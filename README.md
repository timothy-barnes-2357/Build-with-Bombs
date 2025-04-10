# Build with Bombs
## A Minecraft Java Edition mod for procedural house generation by diffusion

This repo contains two components:

- "inference_dll" This contains a C++ DLL that sets up CUDA and calls TensorRT. It provides a number of exported functions for use by Java.
- "mod_neoforge" This is the Java mod code that calls the inference.dll functions. It handles getting / setting blocks in Minecraft.

Version requirements:
- neoforge-21.1.77
- Minecraft-1.21.1
- TensorRT-10.5.0.18
- CUDA 12.6

TensorRT 10.5 requires a GPU with compute capability >= 7.5. See the [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1050/support-matrix/index.html). Check this Wikipedia table to find the compute capability of your GPU: [Compute capability, GPU semiconductors and Nvidia GPU board products](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

## Social

[buildwithbombs.com](buildwithbombs.com)

[Discord to ask questions](https://discord.gg/2ym2tUV5E3)

Join this server to try it out: `mc.buildwithbombs.com`
