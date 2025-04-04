package tbarnes.diffusionmod;

import net.neoforged.neoforge.event.entity.player.PlayerInteractEvent;
import net.neoforged.neoforge.event.tick.PlayerTickEvent;
import org.slf4j.Logger;
import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.food.FoodProperties;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.CreativeModeTabs;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.minecraft.world.level.material.MapColor;
import net.neoforged.api.distmarker.Dist;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.common.EventBusSubscriber;
import net.neoforged.fml.common.Mod;
import net.neoforged.fml.config.ModConfig;
import net.neoforged.fml.event.lifecycle.FMLClientSetupEvent;
import net.neoforged.fml.event.lifecycle.FMLCommonSetupEvent;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.BuildCreativeModeTabContentsEvent;
import net.neoforged.neoforge.event.server.ServerStartingEvent;
import net.neoforged.neoforge.registries.DeferredBlock;
import net.neoforged.neoforge.registries.DeferredHolder;
import net.neoforged.neoforge.registries.DeferredItem;
import net.neoforged.neoforge.registries.DeferredRegister;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.item.context.UseOnContext;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.core.BlockPos;
import net.neoforged.neoforge.event.entity.player.PlayerEvent;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.ItemStack;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.component.DataComponentType;
import net.minecraft.sounds.SoundEvents;
import net.neoforged.neoforge.event.level.BlockEvent;
import net.minecraft.world.level.Level.ExplosionInteraction;
import net.minecraft.world.entity.item.PrimedTnt;
import net.neoforged.neoforge.event.tick.LevelTickEvent;
import net.minecraft.world.level.block.SlabBlock;
import net.minecraft.world.level.block.state.properties.SlabType;
import net.minecraft.world.level.block.state.properties.*;
import java.util.*;
import java.util.function.Supplier;

@Mod(DiffusionMod.MODID)
public class DiffusionMod {

    public static final String MODID = "diffusionmod";
    private static final Logger LOGGER = LogUtils.getLogger();

    private final Inference infer;
    
    private static final Queue<PrimedTnt> diffusionTnts = new LinkedList<>();
    private static final Component diffusion_tnt_name = Component.literal("Diffusion TNT");

    // Arbitrarily large fuse so we can handle the explosion manually in 
    private static final int maxFuse = 1000; 
    private static final int fuseLength = 80;

    private static BlockPos userClickedPos = new BlockPos(0, 0, 0);
    private static Boolean isDenoising = false;
    private static int denoiseCount = 0;

    private static Boolean startedInit = false;
    private static Boolean completedInit = false;
    private static Boolean startedDiffusion = false;
    private static int previousTimestep = 1000;

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
            // if another embedding is generated)
            Blocks.STONE_BRICKS.defaultBlockState()
    };

    private static final HashMap<String, Integer> block_mapping = new HashMap<>();

    static {
        block_mapping.put("air", 0);                   // air -> air
        block_mapping.put("dirt", 1);                  // dirt -> dirt
        block_mapping.put("white concrete", 2);        // white_concrete -> white_concrete
        block_mapping.put("oak planks", 3);            // oak_planks -> oak_planks
        block_mapping.put("stone bricks", 4);          // stone_bricks -> stone_bricks
        block_mapping.put("grass block", 5);           // grass_block -> grass_block
        block_mapping.put("stone brick slab", 6);      // stone_brick_slab -> stone_brick_slab (bottom default)
        block_mapping.put("glass", 8);                 // glass -> glass 
        block_mapping.put("bookshelf", 10);            // bookshelf -> bookshelf 
        block_mapping.put("gravel", 11);               // gravel -> gravel 
        block_mapping.put("green concrete", 12);       // green_concrete -> green_concrete
        block_mapping.put("oak slab", 13);             // oak_slab -> oak_slab 
        block_mapping.put("sandstone", 14);             // oak_slab -> oak_slab 
        block_mapping.put("cobblestone slab", 6);      // cobblestone_slab -> stone_brick_slab (bottom default)
        block_mapping.put("end stone brick slab", 6);  // end_stone_brick_slab -> stone_brick_slab (bottom default)
        block_mapping.put("spruce planks", 3);         // spruce_planks -> oak_planks
        block_mapping.put("chiseled quartz block", 2); // chiseled_quartz_block -> white_concrete
        block_mapping.put("stone", 4);                 // stone -> stone_bricks
        block_mapping.put("end stone bricks", 4);      // end_stone_bricks -> stone_bricks
        block_mapping.put("cobblestone", 4);           // cobblestone -> stone_bricks
        block_mapping.put("green wool", 5);            // green_wool -> grass_block
        block_mapping.put("stripped oak wood", 3);     // stripped_oak_wood -> oak_planks
        block_mapping.put("granite", 11);              // granite -> gravel
        block_mapping.put("smooth stone", 4);          // smooth_stone -> stone_bricks
        block_mapping.put("brick slab", 6);            // brick_slab -> stone_brick_slab (bottom default)
        block_mapping.put("blackstone slab", 6);       // blackstone_slab -> stone_brick_slab (bottom default)
        block_mapping.put("purpur pillar", 2);         // purpur_pillar -> white_concrete
        block_mapping.put("oak log", 3);               // oak_log -> oak_planks
        block_mapping.put("red nether bricks", 4);     // red_nether_bricks -> stone_bricks
        block_mapping.put("purpur block", 2);          // purpur_block -> white_concrete
        block_mapping.put("birch planks", 3);          // birch_planks -> oak_planks
        block_mapping.put("white wool", 2);            // white_wool -> white_concrete
        block_mapping.put("stripped spruce wood", 3);  // stripped_spruce_wood -> oak_planks
        block_mapping.put("crimson planks", 3);        // crimson_planks -> oak_planks
        block_mapping.put("glass pane", 8);            // glass_pane -> glass
        block_mapping.put("coarse dirt", 1);           // coarse_dirt -> dirt
        block_mapping.put("ancient debris", 2);        // ancient_debris -> white_concrete
        block_mapping.put("jungle planks", 3);         // jungle_planks -> oak_planks
        block_mapping.put("bricks", 4);                // bricks -> stone_bricks
        block_mapping.put("dark oak planks", 3);       // dark_oak_planks -> oak_planks
        block_mapping.put("green concrete powder", 12);// green_concrete_powder -> green_concrete
        block_mapping.put("andesite", 4);              // andesite -> stone_bricks
    }

    /** @brief Constructor
     */
    public DiffusionMod(IEventBus modEventBus, ModContainer modContainer) {

        infer = new Inference();

        NeoForge.EVENT_BUS.register(this);
        modContainer.registerConfig(ModConfig.Type.COMMON, Config.SPEC);
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
            diffusionTnts.add(tnt);
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

        /* 
         * Handle error reporting from the DLL 
         */
        int lastError = infer.getLastError();

        if (lastError != 0) {
            printMessageToAllPlayers(level, "Denoise model error (" + lastError + ")");
        }

        /* 
         * Handle the logic for diffusion setups. 
         */
        if (isDenoising) {
            /*
             * Logic for receiving new denoised blocks 
             * for each timestep.
             */
            int timestep = infer.getCurrentTimestep();

            if (timestep < previousTimestep) {
                infer.cacheCurrentTimestepForReading();

                for (int x = 0; x < 14; x++) {
                    for (int y = 0; y < 14; y++) {
                        for (int z = 0; z < 14; z++) {

                            int new_id = infer.readBlockFromCachedTimestep(x, y, z);

                            BlockPos position = new BlockPos(
                                    userClickedPos.getX() + x,
                                    userClickedPos.getY() + y,
                                    userClickedPos.getZ() + z);

                            BlockState state = BLOCK_STATES[new_id];

                            level.setBlockAndUpdate(position, state);
                        }
                    }
                }
            }

            if (timestep == 0) {
                isDenoising = false;
                startedDiffusion = false;
                previousTimestep = 1000; // Timesteps start at 1000
            }
        }

        /*
         * This iterates over the active diffusion TNT entities in the world
         * and determines which one(s) should trigger the next diffusion event.
         */
        Iterator<PrimedTnt> iterator = diffusionTnts.iterator();
        Boolean startDiffusion = false;

        while (iterator.hasNext()) {
            PrimedTnt tnt = iterator.next();
            BlockPos pos = tnt.getOnPos();

            int fuseTriggerPoint = maxFuse - fuseLength;

            if (tnt.getFuse() < fuseTriggerPoint) {
                if (!isDenoising) {

                    /* Center the new diffusion on the TNT block */
                    userClickedPos = new BlockPos(
                            tnt.getBlockX() - 7,
                            tnt.getBlockY() - 1,
                            tnt.getBlockZ() - 7);

                    iterator.remove();
                    tnt.discard();

                    isDenoising = true;
                    startDiffusion = true;

                    break; // Break from the while loop since we only start one at a time.
                } else {
                    tnt.setFuse(fuseTriggerPoint);
                }
            }
        }

        if (startDiffusion) {
            /*
             * Context setup
             * Each Minecraft block needs to be converted to a block_id
             * integer that the inference DLL knows how to interpret.
             */
            for (int x = 0; x < 16; x++) {
                for (int y = 0; y < 16; y++) {
                    for (int z = 0; z < 16; z++) {

                        BlockPos position = new BlockPos(
                                userClickedPos.getX() + x,
                                userClickedPos.getY() + y,
                                userClickedPos.getZ() + z);

                        BlockState blockState = level.getBlockState(position);
                        Block block = blockState.getBlock();

                        int block_id = 0;

                        if (block == Blocks.STONE_BRICK_SLAB) {

                            SlabType slabType = blockState.getValue(SlabBlock.TYPE);

                            if (slabType == SlabType.BOTTOM) {
                                block_id = 6; // stone_brick_slab[type=bottom]
                            } else if (slabType == SlabType.TOP) {
                                block_id = 7; // stone_brick_slab[type=top]
                            } else if (slabType == SlabType.DOUBLE) {
                                block_id = 9; // stone_brick_slab[type=double]
                            }
                        } else {
                            String name = block.getName().getString().toLowerCase();
                            block_id = block_mapping.getOrDefault(name, 0);
                        }

                        infer.setContextBlock(x, y, z, block_id);
                    }
                }
            }

            /* Tell the inference DLL that we're starting diffusion */
            infer.startDiffusion();
            startedDiffusion = true;
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

            printMessageToAllPlayers(level, "Denoise model version " + major + "." + minor + "." + patch + " init started");

            infer.init();
            startedInit = true;
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
        tnt.set(CUSTOM_NAME_SUPPLIER, diffusion_tnt_name);

        return tnt;
    }

    private boolean isDiffusionTnt(ItemStack stack) {

        Component customName = stack.get(DataComponents.CUSTOM_NAME);
        return customName != null && customName.equals(diffusion_tnt_name);
    }
}

