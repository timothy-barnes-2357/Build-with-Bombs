/* Copyright (C) 2025 Timothy Barnes 
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Lesser Public License for more
 * details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

package com.buildwithbombs;

import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.neoforged.neoforge.common.NeoForge;
import org.slf4j.Logger;
import com.mojang.logging.LogUtils;
import net.minecraft.network.chat.Component;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.event.server.ServerStartingEvent;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.core.BlockPos;
import net.neoforged.neoforge.event.entity.player.PlayerEvent;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.ItemStack;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.component.DataComponentType;
import net.neoforged.neoforge.event.level.BlockEvent;
import net.minecraft.world.entity.item.PrimedTnt;
import net.neoforged.neoforge.event.tick.LevelTickEvent;
import net.minecraft.world.level.block.SlabBlock;
import net.minecraft.world.level.block.state.properties.SlabType;
import net.minecraft.world.level.block.state.properties.*;
import java.util.*;
import java.util.function.Supplier;

@Mod(BuildWithBombs.MODID)
public class BuildWithBombs {

    private class WorkerJob {
        int id;
        int previousTimestep;
        boolean initComplete;
        BlockPos position;
        Level level; // Which level (overworld, ender, the end) this job should run in.
    }

    private class TntQueueItem {
        PrimedTnt tnt;
        Level level;
    }

    //
    // Constants
    //
    public static final String MODID = "buildwithbombs";

    private static final int MOD_MAJOR = 0; // TODO: Figure out how to sync these with gradle.properties mod_version
    private static final int MOD_MINOR = 2;
    private static final int MOD_MATCH = 1;

    private static final int WORKER_COUNT = 8;
    private static final int MAX_PLAYER_TNT_QUEUE = 16;

    // Arbitrarily large fuse so we can handle the explosion manually.
    private static final int MAX_TNT_FUSE = 1000000; 
    private static final int DIFFUSION_TNT_FUSE_LENGTH = 80;

    private static final int CHUNK_WIDTH = 16;

    private static final Component DIFFUSION_TNT_NAME = Component.literal("Diffusion TNT");

    private static final Logger LOGGER = LogUtils.getLogger();
    private final Inference infer;

    //
    // Variables:
    //
    private static boolean startedInit = false;
    private static boolean completedInit = false;
    private static final Set<WorkerJob> workerJobs = new HashSet<>();

    // This is a hashmap relating each of the player's UUIDs to their queue of primed TNT
    // blocks.
    private static final Map<UUID, Queue<TntQueueItem>> playersTntQueues = new HashMap<>();

    //
    // Class methods:
    //
    
    /** @brief Constructor
     */
    public BuildWithBombs(IEventBus modEventBus, ModContainer modContainer) {

        infer = new Inference();

        NeoForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onPlayerJoin(PlayerEvent.PlayerLoggedInEvent event) {
        Player player = event.getEntity();
        Level level = player.level();

        if (level.isClientSide()) { return; }

        ItemStack diffusionTnt = createDiffusionTnt();

        if (!player.getInventory().contains(diffusionTnt)) {
            player.getInventory().add(diffusionTnt);
        }

        MinecraftServer server = level.getServer();

        if (!startedInit && server != null) {
            int major = infer.getVersionMajor();
            int minor = infer.getVersionMinor();
            int patch = infer.getVersionPatch();

            printMessageToAllPlayersInAllDimensions(server,"mod version (" + MOD_MAJOR + "." + MOD_MINOR + "." + MOD_MATCH + ")");
            printMessageToAllPlayersInAllDimensions(server,"Diffusion engine version (" + major + "." + minor + "." + patch + ")");

            if (major != MOD_MAJOR || minor != MOD_MINOR || patch != MOD_MATCH) {
                printMessageToAllPlayersInAllDimensions(server,"ERROR: Mod and diffusion engine don't match! Init failed.");
            } else {
                infer.startInit(WORKER_COUNT);
                startedInit = true;
            }
        }
    }

    /** @brief This function handles the logic for when the player
     * places one of the diffusion TNT blocks
     */
    @SubscribeEvent
    public void onBlockPlace(BlockEvent.EntityPlaceEvent event) {

        if (event.getLevel().isClientSide()) { return; }

        Player player = null;

        if (event.getEntity() instanceof Player) {
            player = (Player) event.getEntity();
        }

        if (player == null) { return; }

        ItemStack placedStack = player.getMainHandItem();

        if (event.getPlacedBlock().getBlock() == Blocks.TNT && 
                isDiffusionTnt(placedStack)) {

            Level level = (Level)event.getLevel();
            BlockPos pos = event.getPos();

            event.setCanceled(true);

            level.setBlock(pos, Blocks.TNT.defaultBlockState(), 11);
            PrimedTnt tnt = new PrimedTnt(level, pos.getX() + 0.5, pos.getY(), pos.getZ() + 0.5, player);

            tnt.setFuse(MAX_TNT_FUSE);

            level.addFreshEntity(tnt);

            // 
            // Add this new TNT to the player's queue
            //
            UUID playerId = player.getUUID();

            Queue<TntQueueItem> queue = playersTntQueues.get(playerId);

            // If this player has nothing queued, create the queue first
            if (queue == null) {
                queue = new LinkedList<>();
                playersTntQueues.put(playerId, queue);
            }

            // Limit the size of each player queue to MAX_PLAYER_TNT_QUEUE.
            if (queue.size() < MAX_PLAYER_TNT_QUEUE) {

                TntQueueItem item = new TntQueueItem();
                item.tnt = tnt;
                item.level = level;

                queue.add(item);
            } else {
                tnt.discard();
                String error = "Player max TNT queue is " + MAX_PLAYER_TNT_QUEUE;
                player.sendSystemMessage(Component.literal(error));
            }
        }
    }

    /** @brief This is the main logic for the mod. It keeps track of which
     * TNT blocks have been placed and triggers the diffusion process.
     * This onLevelTick function runs once every tick for each dimension 
     * (overworld, nether, the end).
     */
    @SubscribeEvent
    public void onLevelTick(LevelTickEvent.Post event) {

        Level level = event.getLevel();

        if (level.isClientSide()) return;

        // 
        // Run the "runOncePerTick()" function once by per tick by 
        // only triggering on the overworld update.
        //
        ResourceKey<Level> dimension = level.dimension();
        boolean isOverworld = dimension.equals(Level.OVERWORLD);

        if (isOverworld) {
            runOncePerTick(level.getServer()); 
        }

        //
        // Iterate over the active diffusion jobs
        //
        Iterator<WorkerJob> iterator = workerJobs.iterator();

        while (iterator.hasNext()) {

            WorkerJob job = iterator.next();

            if (job.level != level) {
                continue;
            }

            if (job.initComplete) {
                /*
                 * Logic for receiving new denoised blocks 
                 * for each timestep.
                 */
                int timestep = infer.getCurrentTimestep(job.id);

                if (timestep < job.previousTimestep) {

                    infer.cacheCurrentTimestepForReading(job.id);

                    for (int x = 0; x < 14; x++) {
                        for (int y = 0; y < 14; y++) {
                            for (int z = 0; z < 14; z++) {

                                int new_id = infer.readBlockFromCachedTimestep(x, y, z);

                                BlockPos position = new BlockPos(
                                        job.position.getX() + x,
                                        job.position.getY() + y,
                                        job.position.getZ() + z);

                                BlockState state = BLOCK_STATES[new_id];

                                job.level.setBlockAndUpdate(position, state);
                            }
                        }
                    }
                    
                    // Since the world can change while diffusing, we need to update
                    // the model's context continuously.
                    updateDiffusionContext(job);
                }

                if (timestep == 0) {
                    infer.destroyJob(job.id);
                    iterator.remove();
                }

            } else { // if(initComplete)
                updateDiffusionContext(job);

                /* Tell the inference DLL that we're starting diffusion */
                infer.startDiffusion(job.id);
                job.initComplete = true;
            }
        }
    }

    /** @brief Unlike onLevelTick, this function runs once per tick 
     *        instead of once per tick per dimension.
     */
    private void runOncePerTick(MinecraftServer server) {
        /*
         * Handle checking for initialization complete
         */
        if (startedInit && !completedInit) {
            int initComplete = infer.getInitComplete();

            if (initComplete == 1) {
                completedInit = true;
                printMessageToAllPlayersInAllDimensions(server, "Denoise model init complete");
            }
        }

        /* ArrayList
         * Handle error reporting from the DLL
         */
        int lastError = infer.getLastError();

        if (lastError != 0) {
            printMessageToAllPlayersInAllDimensions(server, "Denoise model error (" + lastError + ")");
        }

        // This is an annoying process of converting the hashmap to a list so we can
        // shuffle it to get N random keys. We're doing this to allow all players to
        // an equal chance of having their diffusion TNT blocks assigned to a job.
        List<Map.Entry<UUID, Queue<TntQueueItem>>> shuffled = new ArrayList<>(playersTntQueues.entrySet());
        Collections.shuffle(shuffled);

        /*
         * This iterates over the active diffusion TNT entities in the world
         * and determines which one(s) should trigger the next diffusion event.
         */
        for (Map.Entry<UUID, Queue<TntQueueItem>> entry : shuffled) {

            UUID playerId = entry.getKey();
            Queue<TntQueueItem> tntQueue = entry.getValue();

            TntQueueItem queueItem = tntQueue.peek(); // Just peek, not poll, since we don't know if
            // a job is available yet.

            int fuseTriggerPoint = MAX_TNT_FUSE - DIFFUSION_TNT_FUSE_LENGTH;
            boolean fuseHasRunOut = queueItem.tnt.getFuse() < fuseTriggerPoint;

            if (fuseHasRunOut) {
                int jobId = infer.createJob();

                if (jobId == -1) {
                    // There's nothing to do. There are no workers available so we can't
                    // create another job at the moment. At this point, we break so on
                    // another tick hopefully a worker becomes available.
                    break;
                }

                //
                // Create a new worker job
                //
                WorkerJob job = new WorkerJob();

                job.id = jobId;
                job.previousTimestep = 1000;
                job.position = new BlockPos(
                        queueItem.tnt.getBlockX() - 7, // Center the 14x14x14 chunk on X and Z.
                        queueItem.tnt.getBlockY() - 1,
                        queueItem.tnt.getBlockZ() - 7);
                job.level = queueItem.level;

                workerJobs.add(job);

                //
                // Server item and queue cleanup: 
                //

                // Remove this tnt from the level it was placed in 
                queueItem.tnt.discard();   

                // We now have a valid job ID so we can remove the tnt block from
                // the player's queue and start diffusion.
                tntQueue.poll(); // Remove this tnt from the player's queue

                // Remove the player queue object itself if it's empty.
                if (tntQueue.isEmpty()) {
                    playersTntQueues.remove(playerId);
                }
            }
        }
    }

    /** @brief Update the blocks that are visible to the diffusion model
     */
    private void updateDiffusionContext(WorkerJob job) {
        /*
         * Context setup
         * Each Minecraft block needs to be converted to a block_id
         * integer that the inference DLL knows how to interpret.
         */
        for (int x = 0; x < CHUNK_WIDTH; x++) {
            for (int y = 0; y < CHUNK_WIDTH; y++) {
                for (int z = 0; z < CHUNK_WIDTH; z++) {

                    BlockPos position = new BlockPos(
                            job.position.getX() + x,
                            job.position.getY() + y,
                            job.position.getZ() + z);

                    BlockState blockState = job.level.getBlockState(position);
                    Block block = blockState.getBlock();

                    int blockId = 0;

                    if (block == Blocks.STONE_BRICK_SLAB) {

                        SlabType slabType = blockState.getValue(SlabBlock.TYPE);

                        if (slabType == SlabType.BOTTOM) {
                            blockId = 6; // stone_brick_slab[type=bottom]
                        } else if (slabType == SlabType.TOP) {
                            blockId = 7; // stone_brick_slab[type=top]
                        } else if (slabType == SlabType.DOUBLE) {
                            blockId = 9; // stone_brick_slab[type=double]
                        }
                    } else {
                        String name = block.getName().getString().toLowerCase();
                        blockId = BLOCK_MAPPING.getOrDefault(name, 0);
                    }

                    infer.setContextBlock(job.id, x, y, z, blockId);
                }
            }
        }
    }

    private static void printMessageToAllPlayersInAllDimensions(MinecraftServer server, String message) {

        Component messageComponent = Component.literal(message);

        for (ServerLevel level : server.getAllLevels()) {
            for (Player player : level.players()) {
                player.sendSystemMessage(messageComponent);
            }
        }

        LOGGER.info(message);
    }

    /* We keep track of which TNT blocks are related to this mod by adding a custom name */
    private static ItemStack createDiffusionTnt() {

        ItemStack tnt = new ItemStack(Items.TNT);

        Supplier<DataComponentType<Component>> CUSTOM_NAME_SUPPLIER = () -> DataComponents.CUSTOM_NAME;
        tnt.set(CUSTOM_NAME_SUPPLIER, DIFFUSION_TNT_NAME);

        return tnt;
    }

    private boolean isDiffusionTnt(ItemStack stack) {

        Component customName = stack.get(DataComponents.CUSTOM_NAME);
        return customName != null && customName.equals(DIFFUSION_TNT_NAME);
    }

    //
    // Block lookup tables:
    //
    private static final BlockState[] BLOCK_STATES = new BlockState[] {
            // 0 - minecraft:air
            Blocks.AIR.defaultBlockState(),

            // 1 - minecraft:dirt
            Blocks.DIRT.defaultBlockState(),

            // 2 - minecraft:white_concrete
            Blocks.WHITE_CONCRETE.defaultBlockState(),

            // 3 - minecraft:oak_planks
            Blocks.OAK_PLANKS.defaultBlockState(),

            // 4 - minecraft:stone_bricks
            Blocks.STONE_BRICKS.defaultBlockState(),

            // 5 - minecraft:grass_block
            Blocks.GRASS_BLOCK.defaultBlockState(),

            // 6 - minecraft:stone_brick_slab[type=bottom]
            Blocks.STONE_BRICK_SLAB.defaultBlockState()
                    .setValue(BlockStateProperties.SLAB_TYPE, SlabType.BOTTOM),

            // 7 - minecraft:stone_brick_slab[type=top]
            Blocks.STONE_BRICK_SLAB.defaultBlockState()
                    .setValue(BlockStateProperties.SLAB_TYPE, SlabType.TOP),

            // 8 - minecraft:glass
            Blocks.GLASS.defaultBlockState(),

            // 9 - minecraft:stone_brick_slab[type=double]
            Blocks.STONE_BRICK_SLAB.defaultBlockState()
                    .setValue(BlockStateProperties.SLAB_TYPE, SlabType.DOUBLE),

            // 10 - minecraft:bookshelf
            Blocks.BOOKSHELF.defaultBlockState(),

            // 11 - minecraft:gravel
            Blocks.GRAVEL.defaultBlockState(),

            // 12 - minecraft:green_concrete
            Blocks.GREEN_CONCRETE.defaultBlockState(),

            // 13 - minecraft:oak_slab
            Blocks.OAK_SLAB.defaultBlockState(),

            // 14 - minecraft:sandstone
            Blocks.SANDSTONE.defaultBlockState(),

            // 15 - minecraft:stone_bricks 
            // (same as #4. This is an accidental redundancy. It will be cleaned up
            // if/when another embedding is generated)
            Blocks.STONE_BRICKS.defaultBlockState()
    };

    private static final HashMap<String, Integer> BLOCK_MAPPING = new HashMap<>();

    static {
        BLOCK_MAPPING.put("air", 0);
        BLOCK_MAPPING.put("dirt", 1);
        BLOCK_MAPPING.put("white concrete", 2);
        BLOCK_MAPPING.put("oak planks", 3);
        BLOCK_MAPPING.put("stone bricks", 4);
        BLOCK_MAPPING.put("grass block", 5);
        BLOCK_MAPPING.put("stone brick slab", 6);
        BLOCK_MAPPING.put("glass", 8);
        BLOCK_MAPPING.put("bookshelf", 10);
        BLOCK_MAPPING.put("gravel", 11);
        BLOCK_MAPPING.put("green concrete", 12);
        BLOCK_MAPPING.put("oak slab", 13);
        BLOCK_MAPPING.put("sandstone", 14);
        BLOCK_MAPPING.put("cobblestone slab", 6);
        BLOCK_MAPPING.put("end stone brick slab", 6);
        BLOCK_MAPPING.put("spruce planks", 3);
        BLOCK_MAPPING.put("chiseled quartz block", 2);
        BLOCK_MAPPING.put("stone", 4);
        BLOCK_MAPPING.put("end stone bricks", 4);
        BLOCK_MAPPING.put("cobblestone", 4);
        BLOCK_MAPPING.put("green wool", 5);
        BLOCK_MAPPING.put("stripped oak wood", 3);
        BLOCK_MAPPING.put("granite", 11);
        BLOCK_MAPPING.put("smooth stone", 4);
        BLOCK_MAPPING.put("brick slab", 6);
        BLOCK_MAPPING.put("blackstone slab", 6);
        BLOCK_MAPPING.put("purpur pillar", 2);
        BLOCK_MAPPING.put("oak log", 3);
        BLOCK_MAPPING.put("red nether bricks", 4);
        BLOCK_MAPPING.put("purpur block", 2);
        BLOCK_MAPPING.put("birch planks", 3);
        BLOCK_MAPPING.put("white wool", 2);
        BLOCK_MAPPING.put("stripped spruce wood", 3);
        BLOCK_MAPPING.put("crimson planks", 3);
        BLOCK_MAPPING.put("glass pane", 8);
        BLOCK_MAPPING.put("coarse dirt", 1);
        BLOCK_MAPPING.put("ancient debris", 2);
        BLOCK_MAPPING.put("jungle planks", 3);
        BLOCK_MAPPING.put("bricks", 4);
        BLOCK_MAPPING.put("dark oak planks", 3);
        BLOCK_MAPPING.put("green concrete powder", 12);
        BLOCK_MAPPING.put("andesite", 4);
    }
}

