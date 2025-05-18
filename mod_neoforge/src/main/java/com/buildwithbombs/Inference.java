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

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

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

        //
        // Figure out if we're on Windows or Linux (or somewhere else that
        // we don't support yet)
        //
        String libName;
        boolean isOsWindows = false;

        if (osName.contains("win")) {
            libName = "inference.dll";
            isOsWindows = true;
        } else if (osName.contains("linux")) {
            libName = "libinference.so";
        } else {
            throw new UnsupportedOperationException("Unsupported operating system: " + osName);
        }

        //
        // If we're on Windows, we have .dlls inside the mod's .jar file to extract.
        // easy mod loading. On Linux, the .so files need to be installed manually.
        //
        if (isOsWindows) {
            String[] windowsDllNames = {
                "inference.dll",
                "cudart64_12.dll",
                "nvinfer_10.dll",
                "nvinfer_builder_resource_10.dll",
                "nvonnxparser_10.dll"
            };

            for (String dllName: windowsDllNames) {
                ExportBinaryFromJar(workingDir, dllName);
            }
        }

        //
        // On both Windows and Linux, extract the model parameter file
        //
        ExportBinaryFromJar(workingDir, "ddim_single_update.onnx");

        //
        // Attempt to load the dynamic library to interface it to Java
        //
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


    private void ExportBinaryFromJar(String workingDir, String binaryName) {
        Path outputPath = Path.of(workingDir, binaryName);

        try (InputStream in = Inference.class.getResourceAsStream(binaryName)) {
            if (in == null) {
                String message = "Binary not found in .jar with name: " + binaryName;
                throw new RuntimeException(message, null);
            }

            Files.copy(in, outputPath);
            System.out.println("Extracted DLL to: " + outputPath);

        } catch (Throwable t) {
            String message = "Failed to extract binary '" + binaryName + "' to working directory: " + workingDir;
            throw new RuntimeException(message, t);
        }
    }
}

