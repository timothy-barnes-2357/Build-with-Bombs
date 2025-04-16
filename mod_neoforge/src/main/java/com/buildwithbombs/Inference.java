package com.buildwithbombs;

public class Inference {

    public native int startInit(int worker_count);
    public native int getInitComplete();
    public native int createJob();
    public native int destroyJob(int job_id);
    public native int setContextBlock(int job_id, int x, int y, int z, int block_id);
    public native int startDiffusion(int job_id);
    public native int getCurrentTimestep(int job_id);
    public native int cacheCurrentTimestepForReading(int job_id);
    public native int readBlockFromCachedTimestep(int x, int y, int z);
    public native int getLastError();
    public native int getVersionMajor();
    public native int getVersionMinor();
    public native int getVersionPatch();

    public Inference() {
        String workingDir = System.getProperty("user.dir");
        String osName = System.getProperty("os.name").toLowerCase();
        String libName;

        if (osName.contains("win")) {
            libName = "inference.dll";
        } else if (osName.contains("linux")) {
            libName = "libinference.so";
        } else {
            throw new UnsupportedOperationException("Unsupported operating system: " + osName);
        }

        String libPath = workingDir + java.io.File.separator + libName;

        try {
            System.load(libPath);
        } catch (Throwable t) {

            String message =
                    "The BuildwithBombs mod couldn't load its AI inference library. This needs to be installed for the mod to work. "
                            + t.getClass().getSimpleName() + " - " + t.getMessage();

            throw new RuntimeException(message, t);
        }
    }
}
