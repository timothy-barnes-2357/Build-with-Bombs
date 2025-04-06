# Build with Bombs
## A Minecraft Java Edition mod for procedural house generation by diffusion

This repo contains two components:

- "inference_dll" This contains a C++ DLL that sets up CUDA and calls TensorRT. It provides a number of exported functions for use by Java.
- "mod_neoforge" This is the Java mod code that calls the inference.dll functions. It handles getting / setting blocks in Minecraft.

[buildwithbombs.com](buildwithbombs.com)

[Discord to ask questions](https://discord.gg/2ym2tUV5E3)

Join this server to try it out: `mc.buildwithbombs.com`
