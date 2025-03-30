package tbarnes.diffusionmod;

public class Inference {

    public native int init();
    public native int getInitComplete();
    public native int setContextBlock(int x, int y, int z, int block_id);
    public native int startDiffusion();
    public native int getCurrentTimestep();
    public native int cacheCurrentTimestepForReading();
    public native int readBlockFromCachedTimestep(int x, int y, int z);
    public native int getLastError();
    public native int getVersionMajor();
    public native int getVersionMinor();
    public native int getVersionPatch();

    static {
        String workingDir = System.getProperty("user.dir");
        String osName = System.getProperty("os.name").toLowerCase();
        String libName;
        String libPath;

        if (osName.contains("win")) {
            libName = "inference.dll";
        } else if (osName.contains("linux")) {
            libName = "libinference.so";
        } else {
            throw new UnsupportedOperationException("Unsupported operating system: " + osName);
        }

        libPath = workingDir + java.io.File.separator + libName;
        System.load(libPath);
    }
}