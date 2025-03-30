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

// The value here should match an entry in the META-INF/neoforge.mods.toml file
@Mod(DiffusionMod.MODID)
public class DiffusionMod
{
    // Define mod id in a common place for everything to reference
    public static final String MODID = "diffusionmod";
    // Directly reference a slf4j logger
    private static final Logger LOGGER = LogUtils.getLogger();
    // Create a Deferred Register to hold Blocks which will all be registered under the "examplemod" namespace
    public static final DeferredRegister.Blocks BLOCKS = DeferredRegister.createBlocks(MODID);
    // Create a Deferred Register to hold Items which will all be registered under the "examplemod" namespace
    public static final DeferredRegister.Items ITEMS = DeferredRegister.createItems(MODID);
    // Create a Deferred Register to hold CreativeModeTabs which will all be registered under the "examplemod" namespace
    public static final DeferredRegister<CreativeModeTab> CREATIVE_MODE_TABS = DeferredRegister.create(Registries.CREATIVE_MODE_TAB, MODID);

    // Creates a new Block with the id "examplemod:example_block", combining the namespace and path
    public static final DeferredBlock<Block> EXAMPLE_BLOCK = BLOCKS.registerSimpleBlock("example_block", BlockBehaviour.Properties.of().mapColor(MapColor.STONE));
    // Creates a new BlockItem with the id "examplemod:example_block", combining the namespace and path
    public static final DeferredItem<BlockItem> EXAMPLE_BLOCK_ITEM = ITEMS.registerSimpleBlockItem("example_block", EXAMPLE_BLOCK);

    // Creates a new food item with the id "examplemod:example_id", nutrition 1 and saturation 2
    public static final DeferredItem<Item> EXAMPLE_ITEM = ITEMS.registerSimpleItem("example_item", new Item.Properties().food(new FoodProperties.Builder()
            .alwaysEdible().nutrition(1).saturationModifier(2f).build()));

    public static final BlockState[] BLOCK_STATES = new BlockState[] {
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

            // 15 - minecraft:stone_brick (should be stone_bricks)
            Blocks.STONE_BRICKS.defaultBlockState()
    };

    Inference infer = new Inference();

    static BlockPos userClickedPos = new BlockPos(0, 0, 0);
    static Boolean isDenoising = false;
    static int denoiseCount = 0;

    static Boolean startedInit = false;
    static Boolean completedInit = false;
    static Boolean startedDiffusion = false;
    static int previousTimestep = 1000;

    public static final HashMap<String, Integer> BLOCK_MAPPING = new HashMap<>();

    static {
        BLOCK_MAPPING.put("air", 0);                     // air -> air
        BLOCK_MAPPING.put("dirt", 1);                    // dirt -> dirt
        BLOCK_MAPPING.put("grass block", 5);             // grass_block -> grass_block
        BLOCK_MAPPING.put("cobblestone slab", 6);        // cobblestone_slab -> stone_brick_slab (bottom default)
        BLOCK_MAPPING.put("stone brick slab", 6);        // stone_brick_slab -> stone_brick_slab (bottom default)
        BLOCK_MAPPING.put("end stone brick slab", 6);    // end_stone_brick_slab -> stone_brick_slab (bottom default)
        BLOCK_MAPPING.put("spruce planks", 3);           // spruce_planks -> oak_planks
        BLOCK_MAPPING.put("oak planks", 3);              // oak_planks -> oak_planks
        BLOCK_MAPPING.put("chiseled quartz block", 2);   // chiseled_quartz_block -> white_concrete
        BLOCK_MAPPING.put("stone", 4);                   // stone -> stone_bricks
        BLOCK_MAPPING.put("end stone bricks", 4);        // end_stone_bricks -> stone_bricks
        BLOCK_MAPPING.put("cobblestone", 4);             // cobblestone -> stone_bricks
        BLOCK_MAPPING.put("green wool", 5);              // green_wool -> grass_block
        BLOCK_MAPPING.put("white concrete", 2);          // white_concrete -> white_concrete
        BLOCK_MAPPING.put("white terracotta", 2);        // white_terracotta -> white_concrete
        BLOCK_MAPPING.put("stripped oak wood", 3);       // stripped_oak_wood -> oak_planks
        BLOCK_MAPPING.put("polished blackstone", 2);     // polished_blackstone -> white_concrete
        BLOCK_MAPPING.put("red sandstone", 2);           // red_sandstone -> white_concrete
        BLOCK_MAPPING.put("granite", 11);                // granite -> gravel
        BLOCK_MAPPING.put("oak leaves", 0);              // oak_leaves -> air (no match)
        BLOCK_MAPPING.put("birch leaves", 0);            // birch_leaves -> air (no match)
        BLOCK_MAPPING.put("smooth stone", 4);            // smooth_stone -> stone_bricks
        BLOCK_MAPPING.put("brick slab", 6);              // brick_slab -> stone_brick_slab (bottom default)
        BLOCK_MAPPING.put("blackstone slab", 6);         // blackstone_slab -> stone_brick_slab (bottom default)
        BLOCK_MAPPING.put("polished blackstone brick slab", 6); // polished_blackstone_brick_slab -> stone_brick_slab
        BLOCK_MAPPING.put("purpur pillar", 2);           // purpur_pillar -> white_concrete
        BLOCK_MAPPING.put("oak log", 3);                 // oak_log -> oak_planks
        BLOCK_MAPPING.put("red nether bricks", 4);       // red_nether_bricks -> stone_bricks
        BLOCK_MAPPING.put("purpur block", 2);            // purpur_block -> white_concrete
        BLOCK_MAPPING.put("stone bricks", 4);            // stone_bricks -> stone_bricks
        BLOCK_MAPPING.put("birch planks", 3);            // birch_planks -> oak_planks
        BLOCK_MAPPING.put("light gray stained glass pane", 8); // light_gray_stained_glass_pane -> glass
        BLOCK_MAPPING.put("white wool", 2);              // white_wool -> white_concrete
        BLOCK_MAPPING.put("dark oak trapdoor", 0);       // dark_oak_trapdoor -> air (no match)
        BLOCK_MAPPING.put("stripped spruce wood", 3);    // stripped_spruce_wood -> oak_planks
        BLOCK_MAPPING.put("crimson planks", 3);          // crimson_planks -> oak_planks
        BLOCK_MAPPING.put("glass pane", 8);              // glass_pane -> glass
        BLOCK_MAPPING.put("coarse dirt", 1);             // coarse_dirt -> dirt
        BLOCK_MAPPING.put("yellow glazed terracotta", 2);// yellow_glazed_terracotta -> white_concrete
        BLOCK_MAPPING.put("green glazed terracotta", 2); // green_glazed_terracotta -> white_concrete
        BLOCK_MAPPING.put("ancient debris", 2);          // ancient_debris -> white_concrete
        BLOCK_MAPPING.put("jungle planks", 3);           // jungle_planks -> oak_planks
        BLOCK_MAPPING.put("dead brain coral block", 2);  // dead_brain_coral_block -> white_concrete
        BLOCK_MAPPING.put("green terracotta", 2);        // green_terracotta -> white_concrete
        BLOCK_MAPPING.put("dead bubble coral block", 2); // dead_bubble_coral_block -> white_concrete
        BLOCK_MAPPING.put("light gray concrete", 2);     // light_gray_concrete -> white_concrete
        BLOCK_MAPPING.put("bricks", 4);                  // bricks -> stone_bricks
        BLOCK_MAPPING.put("white glazed terracotta", 2); // white_glazed_terracotta -> white_concrete
        BLOCK_MAPPING.put("dark oak planks", 3);         // dark_oak_planks -> oak_planks
        BLOCK_MAPPING.put("red sandstone wall", 0);      // red_sandstone_wall -> air (no match)
        BLOCK_MAPPING.put("light gray terracotta", 2);   // light_gray_terracotta -> white_concrete
        BLOCK_MAPPING.put("green concrete", 12);         // green_concrete -> green_concrete
        BLOCK_MAPPING.put("green concrete powder", 12);  // green_concrete_powder -> green_concrete
        BLOCK_MAPPING.put("andesite", 4);                // andesite -> stone_bricks
    }

    public void printMessageToAllPlayers(Level level, String message) {
        Component messageComponent = Component.literal(message);

        for (Player player : level.players()) {
            player.sendSystemMessage(messageComponent);
        }
    }

    @SubscribeEvent
    public void diffusionTick(PlayerTickEvent.Post event) {
        Level level = event.getEntity().level();

        if (startedInit && !completedInit) {
            int initComplete = infer.getInitComplete();

            if (initComplete == 1) {
                completedInit = true;
                printMessageToAllPlayers(level, "Denoise model init complete");
            }
        }

        int lastError = infer.getLastError();

        if (lastError != 0) {
            printMessageToAllPlayers(level, "Denoise model error (" + lastError + ")");
        }

        if (isDenoising) {

            if (!startedInit) {

                int major = infer.getVersionMajor();
                int minor = infer.getVersionMinor();
                int patch = infer.getVersionPatch();

                printMessageToAllPlayers(level, "Denoise model version " + major + "." + minor + "." + patch + " init started");

                infer.init();
                startedInit = true;
            }

            if (!startedDiffusion) {

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
                                block_id = BLOCK_MAPPING.getOrDefault(name, 0);
                            }

                            infer.setContextBlock(x, y, z, block_id);
                        }
                    }
                }

                infer.startDiffusion();
                startedDiffusion = true;
            }

            int timestep = infer.getCurrentTimestep();

            if (timestep < previousTimestep) {
                infer.cacheCurrentTimestepForReading();

                for (int x = 0; x < 14; x++) {
                    for (int y = 0; y < 14; y++) {
                        for (int z = 0; z < 14; z++) {

                            //int new_id = DUMMY_IDS[x + 14 * y + (14 * 14) * z];
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
                previousTimestep = 1000;
            }
        }
    }

    public static final DeferredItem<Item> DIFFUSION_EGG = ITEMS.register("diffusion_egg", () ->
            new Item(new Item.Properties().stacksTo(16)) {
                @Override
                public InteractionResult useOn(UseOnContext context) {
                    if (!context.getLevel().isClientSide) {
                        BlockPos pos = context.getClickedPos();

                        if (!isDenoising) {
                            userClickedPos = pos;
                            isDenoising = true;
                        }

                        //BlockState currentState = level.getBlockState(pos);
                        //Block block = currentState.getBlock();
                        //BlockState defaultState = block.defaultBlockState();
                        //level.setBlockAndUpdate(pos, defaultState);

                        ItemStack stack = context.getItemInHand();
                        stack.shrink(1);

                        return InteractionResult.SUCCESS;
                    }
                    return InteractionResult.CONSUME;
                }
            }
    );

    // Add this to include the egg in your creative tab
    public static final DeferredHolder<CreativeModeTab, CreativeModeTab> EXAMPLE_TAB = CREATIVE_MODE_TABS.register("example_tab", () -> CreativeModeTab.builder()
            .title(Component.translatable("itemGroup.examplemod"))
            .withTabsBefore(CreativeModeTabs.COMBAT)
            .icon(() -> EXAMPLE_ITEM.get().getDefaultInstance())
            .displayItems((parameters, output) -> {
                output.accept(EXAMPLE_ITEM.get());
                output.accept(DIFFUSION_EGG.get());
            }).build());

    // The constructor for the mod class is the first code that is run when your mod is loaded.
    // FML will recognize some parameter types like IEventBus or ModContainer and pass them in automatically.
    public DiffusionMod(IEventBus modEventBus, ModContainer modContainer)
    {
        String file = "Current working directory: " + System.getProperty("user.dir");
        // Register the commonSetup method for modloading
        modEventBus.addListener(this::commonSetup);

        //NeoForge.EVENT_BUS.addListener(DiffusionMod::denoiseTick);

        // Register the Deferred Register to the mod event bus so blocks get registered
        BLOCKS.register(modEventBus);
        // Register the Deferred Register to the mod event bus so items get registered
        ITEMS.register(modEventBus);
        // Register the Deferred Register to the mod event bus so tabs get registered
        CREATIVE_MODE_TABS.register(modEventBus);

        // Register ourselves for server and other game events we are interested in.
        // Note that this is necessary if and only if we want *this* class (ExampleMod) to respond directly to events.
        // Do not add this line if there are no @SubscribeEvent-annotated functions in this class, like onServerStarting() below.
        NeoForge.EVENT_BUS.register(this);

        // Register the item to a creative tab
        modEventBus.addListener(this::addCreative);

        // Register our mod's ModConfigSpec so that FML can create and load the config file for us
        modContainer.registerConfig(ModConfig.Type.COMMON, Config.SPEC);
    }

    private void commonSetup(final FMLCommonSetupEvent event)
    {
        // Some common setup code
        LOGGER.info("HELLO FROM COMMON SETUP");

        if (Config.logDirtBlock)
            LOGGER.info("DIRT BLOCK >> {}", BuiltInRegistries.BLOCK.getKey(Blocks.DIRT));

        LOGGER.info(Config.magicNumberIntroduction + Config.magicNumber);

        Config.items.forEach((item) -> LOGGER.info("ITEM >> {}", item.toString()));
    }

    // Add the example block item to the building blocks tab
    private void addCreative(BuildCreativeModeTabContentsEvent event)
    {
        if (event.getTabKey() == CreativeModeTabs.BUILDING_BLOCKS)
            event.accept(EXAMPLE_BLOCK_ITEM);
    }

    // You can use SubscribeEvent and let the Event Bus discover methods to call
    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event)
    {
        // Do something when the server starts
        LOGGER.info("HELLO from server starting");
    }

    /// Begin new TNT ///////////////////////////////////////////////////////////
    ///
    ///
    private static final Set<PrimedTnt> magicTnts = new HashSet<>();

    public static void markMagicTnt(PrimedTnt tnt) {
        magicTnts.add(tnt);
    }

    public static boolean isMagicTnt(PrimedTnt tnt) {
        return magicTnts.contains(tnt);
    }

    private static final Supplier<DataComponentType<Component>> CUSTOM_NAME_SUPPLIER = () -> DataComponents.CUSTOM_NAME;
    private static final Component EXPLOSION_WOOL_NAME = Component.literal("Explosion Wool").withStyle(style -> style.withColor(0xFF5555));
    private static final Component MAGIC_TNT_NAME = Component.literal("Magic TNT");

    public static ItemStack createMagicTnt() {
        ItemStack tnt = new ItemStack(Items.TNT);
        tnt.set(CUSTOM_NAME_SUPPLIER, MAGIC_TNT_NAME);
        return tnt;
    }

    private boolean isMagicTnt(ItemStack stack) {
        Component customName = stack.get(DataComponents.CUSTOM_NAME);
        return customName != null && customName.equals(MAGIC_TNT_NAME);
    }

    @SubscribeEvent
    public void onEntityTick(LevelTickEvent.Post event) {
        Level level = event.getLevel();

        if (level.isClientSide()) return;

        Iterator<PrimedTnt> iterator = magicTnts.iterator();

        while (iterator.hasNext()) {
            PrimedTnt tnt = iterator.next();
            BlockPos pos = tnt.getOnPos();

            if (tnt.getFuse() < 80) {
                if (!isDenoising) {
                    userClickedPos = new BlockPos(
                            tnt.getBlockX() - 7,
                            tnt.getBlockY() - 1,
                            tnt.getBlockZ() - 7);
                    isDenoising = true;

                    level.playSound(null, pos, SoundEvents.DRAGON_FIREBALL_EXPLODE, net.minecraft.sounds.SoundSource.BLOCKS, 1.0f, 0.25f);

                    iterator.remove();
                    tnt.discard();
                } else {
                    tnt.setFuse(90); // Set the fuse longer to queue for another denoising explosion.
                }
            }
        }
    }

    @SubscribeEvent
    public void onRightClickBlock(PlayerInteractEvent.RightClickBlock event) {
        Player player = event.getEntity();
        Level level = event.getLevel();
        BlockPos pos = event.getPos();
        ItemStack itemStack = event.getItemStack();

        if (level.isClientSide()) return;

        level.playSound(null, pos.getX(), pos.getY(), pos.getZ(), SoundEvents.SLIME_DEATH, net.minecraft.sounds.SoundSource.BLOCKS, 1.0F, 1.0F);
    }

    @SubscribeEvent
    public void onBlockPlace(BlockEvent.EntityPlaceEvent event) {
        if (event.getLevel().isClientSide()) return;

        Player player = event.getEntity() instanceof Player ? (Player) event.getEntity() : null;
        if (player == null) return;

        ItemStack placedStack = player.getMainHandItem();
        if (event.getPlacedBlock().getBlock() == Blocks.TNT && isMagicTnt(placedStack)) {

            Level level = (Level) event.getLevel();
            BlockPos pos = event.getPos();

            // Cancel default TNT placement
            event.setCanceled(true);

            // Place vanilla TNT and prime it
            level.setBlock(pos, Blocks.TNT.defaultBlockState(), 11);
            PrimedTnt tnt = new PrimedTnt(level, pos.getX() + 0.5, pos.getY(), pos.getZ() + 0.5, player);
            tnt.setFuse(160);
            level.addFreshEntity(tnt);

            level.playSound(null, pos, SoundEvents.TNT_PRIMED, net.minecraft.sounds.SoundSource.BLOCKS, 1.0f, 1.0f);


            // Mark this TNT as "magic" using a server-side tracking system
            markMagicTnt(tnt);

        } else if (placedStack.getItem() == Items.WHITE_WOOL && isExplosionWool(placedStack)) {
            Level level = (Level) event.getLevel();
            BlockPos pos = event.getPos();

            event.setCanceled(true);

            level.explode(
                    null,
                    pos.getX() + 0.5, pos.getY() + 0.5, pos.getZ() + 0.5,
                    4.0f,
                    ExplosionInteraction.BLOCK
            );

            level.playSound(null, pos, SoundEvents.TNT_PRIMED, net.minecraft.sounds.SoundSource.BLOCKS, 1.0f, 1.0f);
            //level.playSound(null, pos, SoundEvents.GENERIC_EXPLODE, net.minecraft.sounds.SoundSource.BLOCKS, 1.0f, 1.0f);

            LOGGER.info("Explosion Wool detonated at " + pos + " by " + player.getName().getString());
        }
    }

    /// Begin Explosion Wool///////////////////////////////////////////////////////////
    ///


    private boolean isExplosionWool(ItemStack stack) {
        Component customName = stack.get(DataComponents.CUSTOM_NAME);
        return customName != null && customName.equals(EXPLOSION_WOOL_NAME);
    }


    public static ItemStack createExplosionWool() {
        ItemStack wool = new ItemStack(Items.WHITE_WOOL);
        wool.set(CUSTOM_NAME_SUPPLIER, EXPLOSION_WOOL_NAME);
        return wool;
    }

    @SubscribeEvent
    public void onPlayerJoin(PlayerEvent.PlayerLoggedInEvent event) {
        Player player = event.getEntity();
        if (!player.level().isClientSide()) {  // Server-side only
            ItemStack wool = createExplosionWool();

            if (!player.getInventory().contains(wool)) {
                player.getInventory().add(wool);
                LOGGER.info("Gave " + player.getName().getString() + " an Explosion Wool");
            }

            ItemStack specialTnt = createMagicTnt();

            if (!player.getInventory().contains(specialTnt)) {
                player.getInventory().add(specialTnt);
                LOGGER.info("Gave " + player.getName().getString() + " magic tnt");
            }

        }
    }

    /// ///////////////////////////////////////////////////////////

    // You can use EventBusSubscriber to automatically register all static methods in the class annotated with @SubscribeEvent
    @EventBusSubscriber(modid = MODID, bus = EventBusSubscriber.Bus.MOD, value = Dist.CLIENT)
    public static class ClientModEvents
    {
        @SubscribeEvent
        public static void onClientSetup(FMLClientSetupEvent event)
        {
            // Some client setup code
            LOGGER.info("HELLO FROM CLIENT SETUP");
            LOGGER.info("MINECRAFT NAME >> {}", Minecraft.getInstance().getUser().getName());
        }
    }
}
