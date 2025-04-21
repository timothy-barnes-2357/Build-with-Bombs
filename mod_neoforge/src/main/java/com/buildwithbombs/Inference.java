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
