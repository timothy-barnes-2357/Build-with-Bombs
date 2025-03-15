package tbarnes.diffusionmod;

public class Inference {

    public native int init();
    public native int setContextBlock(int x, int y, int z, int block_id);
    public native int startDiffusion();
    public native int getCurrentTimestep();
    public native int cacheCurrentTimestepForReading();
    public native int readBlockFromCachedTimestep(int x, int y, int z);

    static {
        System.load("C:/Users/tbarnes/Desktop/projects/voxel-diffusion-minecraft-mod/inference_dll/visual_studio_build/x64/Release/inference.dll");
    }

}