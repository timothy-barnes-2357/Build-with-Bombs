# Build with Bombs (Creeper Horde experiment) 💣

This branch has code for a game prototype involving waves of creepers that descending on the player. It's a gameplay experiment to see if building structures to fend off creepers makes for an interesting game.

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

## Social

[buildwithbombs.com](https://buildwithbombs.com)

[Discord to ask questions](https://discord.gg/2ym2tUV5E3)

Join this server to try it out (no client-side mod required): `mc.buildwithbombs.com` 🧨
