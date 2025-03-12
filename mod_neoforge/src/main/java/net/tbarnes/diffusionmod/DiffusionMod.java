package net.tbarnes.diffusionmod;

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
import net.minecraft.world.item.ItemStack;

import net.minecraft.world.level.block.state.properties.*;


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

            // 3 - minecraft:stone_brick_slab
            Blocks.STONE_BRICK_SLAB.defaultBlockState(),

            // 4 - minecraft:grass_block
            Blocks.GRASS_BLOCK.defaultBlockState(),

            // 5 - minecraft:oak_planks
            Blocks.OAK_PLANKS.defaultBlockState(),

            // 6 - minecraft:stone_bricks
            Blocks.STONE_BRICKS.defaultBlockState(),

            // 7 - minecraft:stripped_oak_wood
            Blocks.STRIPPED_OAK_WOOD.defaultBlockState(),

            // 8 - minecraft:end_stone_bricks
            Blocks.END_STONE_BRICKS.defaultBlockState(),

            // 9 - minecraft:white_wool
            Blocks.WHITE_WOOL.defaultBlockState(),

            // 10 - minecraft:green_concrete
            Blocks.GREEN_CONCRETE.defaultBlockState(),

            // 11 - minecraft:glass_pane[east=true,north=false,south=false,west=true]
            Blocks.AIR.defaultBlockState(),

            // 12 - minecraft:smooth_stone
            Blocks.SMOOTH_STONE.defaultBlockState(),

            // 13 - minecraft:brown_shulker_box
            Blocks.BROWN_SHULKER_BOX.defaultBlockState(),

            // 14 - minecraft:glass_pane[east=false,north=true,south=true,west=false]
            Blocks.AIR.defaultBlockState(),

            // 15 - minecraft:oak_slab
            Blocks.OAK_SLAB.defaultBlockState(),

            // 16 - minecraft:sandstone
            Blocks.SANDSTONE.defaultBlockState(),

            // 17 - minecraft:bricks
            Blocks.BRICKS.defaultBlockState(),

            // 18 - minecraft:stone_brick_stairs[facing=north,half=bottom,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.BOTTOM)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 19 - minecraft:stone_brick_stairs[facing=south,half=bottom,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.BOTTOM)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 20 - minecraft:stone_brick_stairs[facing=east,half=bottom,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.BOTTOM)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 21 - minecraft:stone_brick_stairs[facing=west,half=bottom,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.BOTTOM)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 22 - minecraft:stone_brick
            Blocks.STONE_BRICKS.defaultBlockState(),

            // 23 - minecraft:bookshelf
            Blocks.BOOKSHELF.defaultBlockState(),

            // 24 - minecraft:glass
            Blocks.GLASS.defaultBlockState(),

            // 25 - minecraft:gravel
            Blocks.GRAVEL.defaultBlockState(),

            // 26 - minecraft:stone_brick_stairs[facing=south,half=top,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.TOP)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 27 - minecraft:stone_brick_stairs[facing=north,half=top,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.TOP)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 28 - minecraft:stone_brick_stairs[facing=west,half=top,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.TOP)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 29 - minecraft:stone_brick_stairs[facing=east,half=top,shape=straight]
            Blocks.STONE_BRICK_STAIRS.defaultBlockState()
                    .setValue(BlockStateProperties.HALF, Half.TOP)
                    .setValue(BlockStateProperties.STAIRS_SHAPE, StairsShape.STRAIGHT),

            // 30 - minecraft:dropper
            Blocks.DROPPER.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),

            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
            Blocks.AIR.defaultBlockState(),
    };

    public static final int[] DUMMY_IDS = new int[] {
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,10,4,4,4,2,2,2,2,2,4,6,6,6,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,2,2,2,2,2,4,6,6,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,2,2,2,2,2,4,6,6,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,2,2,2,2,2,4,6,6,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,4,4,4,1,1,1,1,1,1,1,6,6,6,2,0,0,0,6,6,6,20,20,6,6,0,0,0,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,29,29,29,0,0,0,0,3,3,3,3,3,3,2,2,2,2,0,0,0,0,0,0,0,0,0,0,3,3,2,3,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,6,6,6,0,0,0,0,6,7,7,7,7,7,6,30,30,30,0,0,0,0,0,0,0,0,0,0,2,30,30,30,0,0,0,0,0,0,0,0,0,0,2,30,30,30,0,0,0,0,0,0,0,0,0,0,2,21,21,21,0,0,0,3,3,3,3,3,3,3,2,2,2,2,0,0,0,0,3,3,3,0,0,0,3,3,8,3,0,0,0,0,0,0,3,3,3,0,3,3,3,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,10,4,4,1,1,1,1,1,1,1,6,6,6,0,0,0,0,6,6,6,7,7,6,6,0,0,0,0,0,0,0,2,74,2,36,36,2,2,0,0,0,0,0,0,0,2,14,2,35,35,2,2,0,0,0,0,0,0,0,2,14,2,29,29,2,2,0,0,0,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,2,2,2,2,2,8,8,8,0,0,0,0,0,0,3,8,2,2,2,8,3,0,0,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,10,4,1,1,1,1,1,1,1,6,6,6,0,0,0,0,6,5,5,5,5,5,6,0,0,0,0,0,0,0,2,18,11,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,8,0,0,0,0,0,0,3,8,8,8,8,8,3,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,4,4,4,1,1,1,1,1,1,1,6,6,6,2,0,0,0,6,5,5,5,5,5,6,0,0,0,2,0,0,0,19,0,11,19,0,0,2,0,0,0,0,0,0,0,11,0,0,0,0,0,2,0,0,0,0,0,0,0,11,0,0,0,0,0,2,0,0,0,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,8,0,0,0,0,0,0,3,8,8,8,8,3,3,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,6,6,6,0,0,0,0,6,5,5,5,5,5,6,0,20,0,0,0,0,0,19,18,66,0,0,0,2,0,0,0,0,0,0,0,11,0,0,0,0,0,2,0,0,0,0,0,0,0,11,0,0,0,0,0,2,0,0,0,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,8,0,0,0,0,0,0,3,8,8,8,8,3,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,1,1,1,0,0,0,0,6,5,5,5,5,5,6,6,7,6,0,0,0,0,2,0,0,0,0,0,2,2,36,2,0,0,0,0,2,0,0,0,0,0,2,2,35,2,0,0,0,0,2,0,0,0,0,0,2,2,2,2,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,3,0,0,0,0,0,0,3,8,8,8,8,3,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,1,1,1,0,0,0,0,6,5,5,5,5,5,0,0,0,6,0,0,0,0,2,0,0,0,53,0,2,0,0,2,0,0,0,0,2,0,0,0,0,0,2,0,0,2,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,3,2,2,2,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,3,0,0,0,0,0,0,3,8,8,8,8,3,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,4,4,4,1,1,1,1,1,1,1,1,1,1,2,0,0,0,6,5,5,5,5,5,0,0,0,6,2,0,0,0,2,0,0,0,19,0,38,0,0,2,0,0,0,0,2,0,0,0,0,0,37,0,0,2,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,3,2,2,0,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,3,0,0,0,0,0,0,3,8,8,8,8,3,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,1,1,1,1,1,1,0,0,0,0,6,5,5,5,5,5,0,0,0,6,0,0,0,0,2,0,0,0,19,0,0,0,0,18,0,0,0,0,2,0,0,0,0,0,0,0,0,11,0,0,0,0,2,0,0,0,0,0,0,0,39,11,0,0,0,3,2,2,0,2,2,2,2,2,2,2,0,0,0,0,3,8,8,8,8,8,8,8,8,3,0,0,0,0,0,0,3,8,8,8,8,3,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    };


    static BlockPos userClickedPos = new BlockPos(0, 0, 0);
    static Boolean isDenoising = false;
    static int denoiseCount = 0;

    @SubscribeEvent
    public void diffusionTick(PlayerTickEvent.Post event) {

        if (isDenoising) {

            if (denoiseCount > 200) {
                isDenoising = false;
                denoiseCount = 0;
            } else {
                Level level = event.getEntity().level();

                for (int x = 0; x < 14; x++) {
                    for (int y = 0; y < 14; y++) {
                        for (int z = 0; z < 14; z++) {

                            int new_id = DUMMY_IDS[x + 14 * y + (14 * 14) * z];

                            BlockPos relative = new BlockPos(
                                    userClickedPos.getX() + x,
                                    userClickedPos.getY() + y + denoiseCount,
                                    userClickedPos.getZ() + z);

                            BlockState state = BLOCK_STATES[new_id];

                            level.setBlockAndUpdate(relative, state);
                        }
                    }
                }

                denoiseCount += 1;
            }
        }
    }

    public static final DeferredItem<Item> DIRT_EGG = ITEMS.register("dirt_egg", () ->
            new Item(new Item.Properties().stacksTo(16)) {
                @Override
                public InteractionResult useOn(UseOnContext context) {
                    if (!context.getLevel().isClientSide) {
                        BlockPos pos = context.getClickedPos();

                        userClickedPos = pos;
                        isDenoising = true;

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
                output.accept(DIRT_EGG.get()); // Add the dirt egg to the tab
            }).build());

    // The constructor for the mod class is the first code that is run when your mod is loaded.
    // FML will recognize some parameter types like IEventBus or ModContainer and pass them in automatically.
    public DiffusionMod(IEventBus modEventBus, ModContainer modContainer)
    {
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
