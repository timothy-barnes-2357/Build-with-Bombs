Build with Bombs: A Minecraft Java Edition mod for procedural house generation by diffusion.

This repo contains two components:

- "inference_dll" This contains a C++ DLL that sets up CUDA and calls TensorRT. It provides a number of exported functions for use by Java.
- "neoforge_mod" This is the Java mod code that calls the inference.dll functions. It handles getting / setting blocks in Minecraft.

Visit buildwithbombs.com
