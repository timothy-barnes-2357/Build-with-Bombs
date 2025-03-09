
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

static IExecutionContext* context;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            printf("%s\n", msg);
    }
} gLogger;


int main() {

    const char* engine_file = "../experiments/TestTensorRT/ddim_single_update_fp16.trt";

    FILE* file = fopen(engine_file, "rb");
    if (!file) {
        printf("Error opening engine file: %s\n", engine_file);
        return 0;
    }

    fseek(file, 0, SEEK_END);
    int engine_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* engine_data = (char*)malloc(engine_size);
    assert(engine_data);

    fread(engine_data, 1, engine_size, file);
    fclose(file);

    printf("Loaded engine\n");

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime);

    printf("Created runtime\n\n");

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data, engine_size);
    free(engine_data);
    assert(engine);

    printf("Finished deserializing CUDA engine\n\n");

    context = engine->createExecutionContext();
    assert(context);

    printf("Finished trt init\n");

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType dtype = engine->getTensorDataType(name);

        // Calculate the size in bytes
        int elementSize = 0;
        switch (dtype) {
        case nvinfer1::DataType::kFLOAT: elementSize = 4; break;
        case nvinfer1::DataType::kHALF: elementSize = 2; break;
        case nvinfer1::DataType::kINT8: elementSize = 1; break;
        case nvinfer1::DataType::kINT32: elementSize = 4; break;
        default: printf("Unknown data type\n"); continue;
        }

        int totalSize = elementSize;
        printf("Name %d = %s, Shape = [", i, name);
        for (int j = 0; j < dims.nbDims; j++) {
            printf("%d", dims.d[j]);
            totalSize *= dims.d[j];
            if (j < dims.nbDims - 1) printf(", ");
        }
        printf("], Size in Bytes = %d\n", totalSize);
    }

    return 0;
}