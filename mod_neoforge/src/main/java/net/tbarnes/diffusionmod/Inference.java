package net.tbarnes.diffusionmod;

public class Inference {
    public native int readBlockFromCachedTimestep(int x, int y, int z);

    static {
        System.load("C:/Users/tbarnes/Desktop/projects/voxel-diffusion-minecraft-mod/inference_dll/visual_studio_build/x64/Release/inference.dll");
    }
}