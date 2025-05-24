# Build with Bombs 💣
## A Minecraft Java mod for procedural house generation by diffusion

This repo contains two components:

- "inference_dll" This contains a C++ DLL that sets up CUDA and calls TensorRT. It provides a number of exported functions for use by Java.
- "mod_neoforge" This is the Java mod code that calls the inference.dll functions. It handles getting / setting blocks in Minecraft.

Version requirements:
- neoforge-21.1.77
- Minecraft-1.21.1
- TensorRT-10.5.0.18
- CUDA 12.6

## Hardware compatibility
TensorRT 10.5 requires an NVIDIA GPU with compute capability >= 7.5. This means it requires an **RTX 2060** or better, a **GTX 1660 Ti** or better, an **MX550** or better, or a **Tesla T4** or better. See the [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1050/support-matrix/index.html). Check this Wikipedia table to find the compute capability of your GPU: [Compute capability, GPU semiconductors and Nvidia GPU board products](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

## Setup Guide
This setup guide includes steps for building the .jar Java mod file as well as building the native executable.

#### Required packages and programs
* CMake: https://cmake.org/download/
* Java 21 JDK: https://www.oracle.com/java/technologies/downloads/#jdk21-windows
* CUDA 12.6: https://developer.nvidia.com/cuda-12-6-0-download-archive
* TensorRT 10.5: https://developer.nvidia.com/tensorrt/download/10x

For Linux, install the package "TensorRT 10.5 GA for Linux x86_64 and CUDA 12.0 to 12.6 TAR Package". Extract the .tar and move the contents to /usr/local `sudo mv TensorRT-10.5.0.18 /usr/local/tensorrt-10.5`

#### Build steps

1. In the mod_neoforge directory, run `./gradlew setup`. This can take some time as it downloads the NeoForge dependencies.

2. Run `./gradlew build`. After a successful build, the mod .jar file will be located in the build folder `mod_neoforge/build/libs/buildwithbombs-0.2.1.jar`. 

3. Build the inference DLL using CMake.
In the `inference_dll` directory, run:
    * `mkdir build`
    * `cd build` 
    * `cmake ..`
    * `cmake --build . --config Release`

4. Copy the newly built library (`inference.dll` on Windows, `libinference.so` on Linux) to the mod's run folder. The `run` folder should have been created after the `./gradlew setup` step.
    * `cp libinference.so ../mod_neoforge/run`
  
5. Copy the .ONNX model file from the GitHub [release page](https://github.com/timothy-barnes-2357/Build-with-Bombs/releases/download/v0.2.1/ddim_single_update.onnx) and place it in the `mod_neoforge/run` directory. This contains the model parameters and must be located next to inference.dll.
  
6. Make sure inference.dll is able to find the TensorRT and CUDA dynamic libraries. Either copy all DLLs into the `mod_neoforge/run` directory, or add the CUDA and TensorRT lib folders to the system path. On Linux, this can be done by `export LD_LIBRARY_PATH=/usr/local/tensorrt-10.5/lib:$LD_LIBRARY_PATH`

7. Test the mod by running `./gradlew runClient`

#### (Optional) Pack native libraries in .jar
1. Copy inference.dll from `inference_dll\build\Release` to `mod_neoforge\native_libraries`.

2. Copy the following NVIDIA DLLs to `mod_neoforge\native_libraries`:
    * cudart64_12.dll (from the install of CUDA)
    * nvinfer_10.dll  (from the install of TensorRT)
    * nvinfer_builder_resource_10.dll (from the install of TensorRT)
    * nvonnxparser_10.dll (from the install of TensorRT)

3. Build the .jar file by running `./gradlew build -Ppack_native_libraries`. This built will take some time. The .jar under `mod_neoforge\build\libs` should be around 1 GB in size. 

4. Verify the .dlls were included by running `jar tf /build/libs/buildwithbombs-*.jar` to see all included files. 

## Social

[buildwithbombs.com](https://buildwithbombs.com)

[Discord to ask questions](https://discord.gg/2ym2tUV5E3)

Join this server to try it out (no client-side mod required): `mc.buildwithbombs.com` 🧨
