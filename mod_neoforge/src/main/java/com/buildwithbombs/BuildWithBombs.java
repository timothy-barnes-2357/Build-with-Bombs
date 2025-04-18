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

import net.minecraft.world.level.chunk.LevelChunk;
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

    public static final String MODID = "buildwithbombs";

    private static final int modMajor = 0; // TODO: Figure out how to sync these with gradle.properties mod_version
    private static final int modMinor = 2;
    private static final int modPatch = 0;
    private static final int chunkWidth = 16;
    private static final int workerCount = 16;
    private static final int maxDiffusionUpdatesPerTick = 8;

    private static final Logger LOGGER = LogUtils.getLogger();

    private final Inference infer;

    // This is a hashmap relating each of the player's UUIDs to their queue of primed TNT
    // blocks.
    private static final Map<UUID, Queue<PrimedTnt>> playerTntQueue = new HashMap<>();

    private class WorkerJob {
        int id;
        int previousTimestep;
        BlockPos position;
        boolean initComplete;
    }

    private static final Set<WorkerJob> workerJobs = new HashSet<>();

    private static final Component diffusionTntName = Component.literal("Diffusion TNT");

    // Arbitrarily large fuse so we can handle the explosion manually.
    private static final int maxFuse = 1000000; 
    private static final int fuseLength = 80;

    private static boolean startedInit = false;
    private static boolean completedInit = false;
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

    private static final HashMap<String, Integer> blockMapping = new HashMap<>();

    static {
        blockMapping.put("air", 0);
        blockMapping.put("dirt", 1);
        blockMapping.put("white concrete", 2);
        blockMapping.put("oak planks", 3);
        blockMapping.put("stone bricks", 4);
        blockMapping.put("grass block", 5);
        blockMapping.put("stone brick slab", 6);
        blockMapping.put("glass", 8);
        blockMapping.put("bookshelf", 10);
        blockMapping.put("gravel", 11);
        blockMapping.put("green concrete", 12);
        blockMapping.put("oak slab", 13);
        blockMapping.put("sandstone", 14);
        blockMapping.put("cobblestone slab", 6);
        blockMapping.put("end stone brick slab", 6);
        blockMapping.put("spruce planks", 3);
        blockMapping.put("chiseled quartz block", 2);
        blockMapping.put("stone", 4);
        blockMapping.put("end stone bricks", 4);
        blockMapping.put("cobblestone", 4);
        blockMapping.put("green wool", 5);
        blockMapping.put("stripped oak wood", 3);
        blockMapping.put("granite", 11);
        blockMapping.put("smooth stone", 4);
        blockMapping.put("brick slab", 6);
        blockMapping.put("blackstone slab", 6);
        blockMapping.put("purpur pillar", 2);
        blockMapping.put("oak log", 3);
        blockMapping.put("red nether bricks", 4);
        blockMapping.put("purpur block", 2);
        blockMapping.put("birch planks", 3);
        blockMapping.put("white wool", 2);
        blockMapping.put("stripped spruce wood", 3);
        blockMapping.put("crimson planks", 3);
        blockMapping.put("glass pane", 8);
        blockMapping.put("coarse dirt", 1);
        blockMapping.put("ancient debris", 2);
        blockMapping.put("jungle planks", 3);
        blockMapping.put("bricks", 4);
        blockMapping.put("dark oak planks", 3);
        blockMapping.put("green concrete powder", 12);
        blockMapping.put("andesite", 4);
    }

    /** @brief Constructor
     */
    public BuildWithBombs(IEventBus modEventBus, ModContainer modContainer) {

        infer = new Inference();

        NeoForge.EVENT_BUS.register(this);
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

            tnt.setFuse(maxFuse);

            level.addFreshEntity(tnt);

            //
            // Add the new diffusion block to the player's queue
            //
            // TODO: Handle logic for limiting the player queue to N items
            UUID playerId = player.getUUID();

            Queue<PrimedTnt> queue = playerTntQueue.get(playerId);

            // If this player has nothing queued, create the queue first
            if (queue == null) {
                queue = new LinkedList<>();
                playerTntQueue.put(playerId, queue);
            }

            queue.add(tnt);
        }
    }

    private void updateDiffusionContext(WorkerJob job, Level level) {
        /*
         * Context setup
         * Each Minecraft block needs to be converted to a block_id
         * integer that the inference DLL knows how to interpret.
         */
        for (int x = 0; x < chunkWidth; x++) {
            for (int y = 0; y < chunkWidth; y++) {
                for (int z = 0; z < chunkWidth; z++) {

                    BlockPos position = new BlockPos(
                            job.position.getX() + x,
                            job.position.getY() + y,
                            job.position.getZ() + z);

                    BlockState blockState = level.getBlockState(position);
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
                        blockId = blockMapping.getOrDefault(name, 0);
                    }

                    infer.setContextBlock(job.id, x, y, z, blockId);
                }
            }
        }
    }

    /** @brief This is the main logic for the mod. It keeps track of which
     * TNT blocks have been placed and triggers the diffusion process.
     */
    @SubscribeEvent
    public void onLevelTick(LevelTickEvent.Post event) {

        Level level = event.getLevel();

        if (level.isClientSide()) return;

        /*
         * Handle checking for initialization complete 
         */
        if (startedInit && !completedInit) {
            int initComplete = infer.getInitComplete();

            if (initComplete == 1) {
                completedInit = true;
                printMessageToAllPlayers(level, "Denoise model init complete");
            }
        }

        /* ArrayList
         * Handle error reporting from the DLL 
         */
        int lastError = infer.getLastError();

        if (lastError != 0) {
            printMessageToAllPlayers(level, "Denoise model error (" + lastError + ")");
        }

        // This is an annoying process of converting the hashmap into a list so we can
        // shuffle it to get N random keys. We're doing this to allow all players to
        // have an equal chance of having their diffusion TNT blocks assigned to a job.
        List<Map.Entry<UUID, Queue<PrimedTnt>>> shuffled = new ArrayList<>(playerTntQueue.entrySet());
        Collections.shuffle(shuffled);

        /*
         * This iterates over the active diffusion TNT entities in the world
         * and determines which one(s) should trigger the next diffusion event.
         */
        for (Map.Entry<UUID, Queue<PrimedTnt>> entry : shuffled) {

            UUID playerId = entry.getKey();
            Queue<PrimedTnt> tntQueue = entry.getValue();

            PrimedTnt tnt = tntQueue.peek(); // Just peek, not poll, since we don't know if
                                              // a job is available yet.

            int fuseTriggerPoint = maxFuse - fuseLength;
            boolean fuseHasRunOut = tnt.getFuse() < fuseTriggerPoint;

            if (fuseHasRunOut) {
                int jobId = infer.createJob();

                if (jobId == -1) {
                    // There's nothing to do. There are no workers available so we can't
                    // create another job at the moment. At this point, we break so on
                    // another tick hopefully a worker becomes available.
                    break;
                }

                // We now have a valid job ID so we can remove the tnt block from
                // the player's queue and start diffusion.
                tntQueue.poll(); // Remove this tnt from the player's queue
                tnt.discard();   // remove this tnt from the level

                // Remove the player queue object itself if it's empty. 
                if (tntQueue.isEmpty()) {
                    playerTntQueue.remove(playerId);
                }

                //
                // Create a new worker job
                //
                WorkerJob job = new WorkerJob();

                job.id = jobId;
                job.previousTimestep = 1000;
                job.position = new BlockPos(
                            tnt.getBlockX() - 7,
                            tnt.getBlockY() - 1,
                            tnt.getBlockZ() - 7);

                workerJobs.add(job);
            }
        }

        int totalDiffusionUpdates = 0;

        // We shuffle the jobs as well to allow reach job to have an equal chance of
        // running per tick if there are more than maxDiffusionUpdatesPerTick number of
        // jobs queued.
        List<WorkerJob> shuffledJobs = new ArrayList<>(workerJobs);
        Collections.shuffle(shuffledJobs);

        Iterator<WorkerJob> iterator = shuffledJobs.iterator();

        //
        // Iterate over the active jobs
        //
        while (iterator.hasNext() &&
                totalDiffusionUpdates < maxDiffusionUpdatesPerTick) {

            WorkerJob job = iterator.next();

            if (job.initComplete) {
                /*
                 * Logic for receiving new denoised blocks 
                 * for each timestep.
                 */
                int timestep = infer.getCurrentTimestep(job.id);

                // Minecraft server behavior or bug:
                // We could supposedly only call level.setBlockAndUpdate when we actually
                // have new data by checking that the current timestep has changed, but in 
                // practice this makes the performance worse instead of better (even though
                // we would be doing less processing in the mod). My hypothesis is that if 
                // I don't call setBlockAndUpdate on a server tick, then the server decides
                // to update the path finding or do other time-intensive tasks. All I know
                // is that the server performance is worse if I don't update these blocks
                // every server tick while the diffusion process is running.
                // (Note written 4-15-2025)

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

                            level.setBlockAndUpdate(position, state);
                        }
                    }
                }

                totalDiffusionUpdates += 1;
                //updateDiffusionContext(job, level);

                if (timestep == 0) {
                    infer.destroyJob(job.id);
                    workerJobs.remove(job);
                }

            } else { // if(initComplete)
                updateDiffusionContext(job, level);

                /* Tell the inference DLL that we're starting diffusion */
                infer.startDiffusion(job.id);
                job.initComplete = true;
            }
        }
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

        if (!startedInit) {
            int major = infer.getVersionMajor();
            int minor = infer.getVersionMinor();
            int patch = infer.getVersionPatch();

            printMessageToAllPlayers(level,"mod version (" + modMajor + "." + modMinor + "." + modPatch + ")");
            printMessageToAllPlayers(level,"Diffusion engine version (" + major + "." + minor + "." + patch + ")");

            if (major != modMajor || minor != modMinor || patch != modPatch) {
                printMessageToAllPlayers(level,"ERROR: Mod and diffusion engine don't match! Init failed.");
            } else {
                infer.startInit(workerCount);
                startedInit = true;
            }
        }
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {
        LOGGER.info("Server starting");

    }

    private static void printMessageToAllPlayers(Level level, String message) {
        Component messageComponent = Component.literal(message);

        for (Player player : level.players()) {
            player.sendSystemMessage(messageComponent);
        }

        LOGGER.info(message);
    }

    /* We keep track of which TNT blocks are related to this mod by adding a custom name */
    private static ItemStack createDiffusionTnt() {

        ItemStack tnt = new ItemStack(Items.TNT);

        Supplier<DataComponentType<Component>> CUSTOM_NAME_SUPPLIER = () -> DataComponents.CUSTOM_NAME;
        tnt.set(CUSTOM_NAME_SUPPLIER, diffusionTntName);

        return tnt;
    }

    private boolean isDiffusionTnt(ItemStack stack) {

        Component customName = stack.get(DataComponents.CUSTOM_NAME);
        return customName != null && customName.equals(diffusionTntName);
    }
}

