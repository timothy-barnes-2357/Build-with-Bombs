
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include <windows.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

/* TODO:
    - Replace asserts with actual error codes
 */

using namespace nvinfer1;

static IExecutionContext* context;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            printf("%s\n", msg);
    }
} gLogger;

#define ID_AIR 0

CRITICAL_SECTION critical_section;

static bool init_complete = false;

const int n_U = 5;
const int n_T = 1000;

const int size_x = 3 * 16 * 16 * 16 * sizeof(float);
const int size_x_context = 3 * 16 * 16 * 16 * sizeof(float);
const int size_x_mask = 1 * 16 * 16 * 16 * sizeof(float);
const int size_normal_epsilon = 3 * 16 * 16 * 16 * sizeof(float);
const int size_normal_z = 3 * 16 * 16 * 16 * sizeof(float);
const int size_alpha = 1000 * sizeof(float);
const int size_alpha_bar = 1000 * sizeof(float);
const int size_beta = 1000 * sizeof(float);

const int instances = 5000;

const float embedding_matrix[15][3] = {
    {     0.0425,      1.4767,      1.7010},
    {    -1.4576,     -1.8709,     -0.0428},
    {     1.8989,     -0.0021,      0.2358},
    {    -0.3358,      0.3482,     -1.1582},
    {    -0.2151,     -1.3739,     -0.9221},
    {    -0.1888,     -1.1246,      0.2749},
    {    -0.6677,     -1.6494,      1.6347},
    {    -0.3955,     -0.4718,      0.7873},
    {    -1.3932,      1.8974,      0.3703},
    {    -0.9238,     -0.4599,      2.4603},
    {     0.4461,      0.3810,     -0.7708},
    {     1.0976,     -0.6826,      0.5087},
    {     1.0425,      2.1311,     -0.3349},
    {    -1.6729,      0.5820,      0.5647},
    {    -1.8158,      0.3562,      0.2172}
};


static float* x_t = NULL;

static float* x_all = NULL;
static float* x_context = NULL;
static float* x_mask = NULL;
static float* normal_epsilon = NULL;
static float* normal_z = NULL;
static float* alpha = NULL;
static float* alpha_bar = NULL;
static float* beta = NULL;

static void* cuda_t = NULL;
static void* cuda_x_t = NULL;
static void* cuda_x_out = NULL;
static void* cuda_x_context = NULL;
static void* cuda_x_mask = NULL;
static void* cuda_normal_epsilon = NULL;
static void* cuda_normal_z = NULL;
static void* cuda_alpha_t = NULL;
static void* cuda_alpha_bar_t = NULL;
static void* cuda_beta_t = NULL;
static void* cuda_beta_t_minus_1 = NULL;
static void* cuda_post_noise_addition = NULL;

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* The caller of this function is responsible for deallocating the memory*/
static float* load_file(const char* filename, int file_size) {

    FILE* file = fopen(filename, "rb");

    if (!file) {
        printf("Cannot find %s\n", filename);
        exit(0);
    }

    fseek(file, 0, SEEK_END);
    int ftell_size = ftell(file);
    rewind(file);

    void* contents = malloc(ftell_size);
    assert(contents);

    size_t bytes_read = fread(contents, 1, ftell_size, file);

    if (bytes_read != file_size) {
        printf("%s size doesn't match (%d != %d)\n", filename, bytes_read, file_size);
        exit(0);
    }

    fclose(file);

    printf("Successfully loaded %s\n", filename);
    return (float*)contents;
}

static void write_to_file(const char* filename, uint8_t* array, size_t size) {
    // Open the file in binary write mode
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write the array to the file
    size_t written = fwrite(array, 1, size, file);
    if (written != size) {
        printf("Error writing to file");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fclose(file);
}



unsigned long __stdcall denoising_main_thread(void *param) {
#if 1
    for (int t = 1000; t >= 0; t -= 1) {
        for (int u = 0; u < n_U; u++) {

            float post_noise_addition = (u < n_U && t > 0) ? 1.0f : 0.0f;
            int load_index = t * n_U + u;

            float* normal_epsilon_t = &normal_epsilon[load_index * (size_normal_epsilon / sizeof(float))];
            float* normal_z_t = &normal_z[load_index * (size_normal_z / sizeof(float))];

            float alpha_t = alpha[t];
            float alpha_bar_t = alpha_bar[t];
            float beta_t = beta[t];
            float beta_t_minus_1 = t > 0 ? beta[t - 1] : 0.0;

            check(cudaMemcpy(cuda_t, &t, sizeof(int32_t), cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_x_t, x_t, size_x, cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_normal_epsilon, normal_epsilon_t, size_normal_epsilon, cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_normal_z, normal_z_t, size_normal_z, cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_alpha_t, &alpha_t, sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_alpha_bar_t, &alpha_bar_t, sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_beta_t, &beta_t, sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_beta_t_minus_1, &beta_t_minus_1, sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(cuda_post_noise_addition, &post_noise_addition, sizeof(float), cudaMemcpyHostToDevice));

            cudaStream_t stream;
            check(cudaStreamCreate(&stream));

            assert(context->enqueueV3(stream));

            check(cudaStreamSynchronize(stream));

            /* TODO: This needs a critical section to avoid modifying the cache function from the other thread. */
            check(cudaMemcpy(x_t, cuda_x_out, size_x, cudaMemcpyDeviceToHost));
        }
    }
#endif

    return 0;
}

extern "C" __declspec(dllexport) 
int __stdcall Java_net_tbarnes_diffusionmod_Inference_init(void* env, void* obj) {

    InitializeCriticalSection(&critical_section);

    const char* engine_file = "../experiments/TestTensorRT/ddim_single_update_fp16.trt";

    FILE* file = fopen(engine_file, "rb");
    if (!file) {
        printf("Error opening engine file: %s\n", engine_file);
        return 1;
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

    x_all = load_file("../experiments/TestTensorRT/save_x_all.bin", instances * size_x);
    x_context = load_file("../experiments/TestTensorRT/save_context.bin", size_x_context);
    x_mask = load_file("../experiments/TestTensorRT/save_mask.bin", size_x_mask);

    normal_epsilon = load_file("../experiments/TestTensorRT/save_normal_epsilon.bin", instances * size_normal_epsilon);
    normal_z = load_file("../experiments/TestTensorRT/save_normal_z.bin", instances * size_normal_z);

    alpha = load_file("../experiments/TestTensorRT/save_alpha.bin", size_alpha);
    alpha_bar = load_file("../experiments/TestTensorRT/save_alpha_bar.bin", size_alpha_bar);
    beta = load_file("../experiments/TestTensorRT/save_beta.bin", size_beta);

    check(cudaMalloc(&cuda_t, sizeof(int32_t)));
    check(cudaMalloc(&cuda_x_t, size_x)); // Allocate only a single instance
    check(cudaMalloc(&cuda_x_out, size_x)); // Output returned by the model
    check(cudaMalloc(&cuda_x_context, size_x_context));
    check(cudaMalloc(&cuda_x_mask, size_x_mask));
    check(cudaMalloc(&cuda_normal_epsilon, size_normal_epsilon));
    check(cudaMalloc(&cuda_normal_z, size_normal_z));
    check(cudaMalloc(&cuda_alpha_t, sizeof(float)));
    check(cudaMalloc(&cuda_alpha_bar_t, sizeof(float)));
    check(cudaMalloc(&cuda_beta_t, sizeof(float)));
    check(cudaMalloc(&cuda_beta_t_minus_1, sizeof(float)));
    check(cudaMalloc(&cuda_post_noise_addition, sizeof(float)));

    assert(context->setTensorAddress("t", cuda_t));
    assert(context->setTensorAddress("x_t", cuda_x_t));
    assert(context->setTensorAddress("x_out", cuda_x_out));
    assert(context->setTensorAddress("context", cuda_x_context));
    assert(context->setTensorAddress("mask", cuda_x_mask));
    assert(context->setTensorAddress("normal_epsilon", cuda_normal_epsilon));
    assert(context->setTensorAddress("normal_z", cuda_normal_z));
    assert(context->setTensorAddress("alpha_t", cuda_alpha_t));
    assert(context->setTensorAddress("alpha_bar_t", cuda_alpha_bar_t));
    assert(context->setTensorAddress("beta_t", cuda_beta_t));
    assert(context->setTensorAddress("beta_t_minus_1", cuda_beta_t_minus_1));
    assert(context->setTensorAddress("post_noise_addition", cuda_post_noise_addition));

    int initial_x_index = (n_T - 1) * n_U;

    init_complete = true;

    return 0;
}

extern "C" __declspec(dllexport)
void __stdcall Java_Inference_setContext(void* env, void* obj, int x, int y, int z, int block_id) {

    /* I need to think through this. I think setting a context block should set the mask, 
       but I need a contextClear function.
     */
    check(cudaMemcpy(cuda_x_context, x_context, size_x_context, cudaMemcpyHostToDevice));
    check(cudaMemcpy(cuda_x_mask, x_mask, size_x_mask, cudaMemcpyHostToDevice));
}

extern "C" __declspec(dllexport)
void __stdcall Java_Inference_startDiffusion(void* env, void* obj) {

    HANDLE thread_handle;
    DWORD thread_id;

    thread_handle = CreateThread(NULL, 0, denoising_main_thread, NULL, 0, &thread_id);
    if (thread_handle == NULL) {
        printf("CreateThread error: %d\n", GetLastError());
        return;
    }
}

static int output_unembed[14][14][14];

extern "C" __declspec(dllexport)
void __stdcall Java_net_tbarnes_diffusionmod_Inference_cacheCurrentTimestepForReading(void* env, void* obj) { // returns timestep that was cashed

    /* Perform matrix multiply of x_t and transpose(embedding_matrix)
     * Since we only care about the index of the largest element in each row of the output
     * 4096x15 matrix, we don't need to actually store the entire matrix, just the largest
     * element in the row. */

    for (int x = 1; x < 15; x++) {
        for (int y = 1; y < 15; y++) {
            for (int z = 1; z < 15; z++) {

                float largest_id_value = -FLT_MAX;
                int largest_id = 0;

                for (int i = 0; i < 15; i++) {
                    float element = 0.0f;

                    for (int j = 0; j < 3; j++) {

                        int x_offset = (j * 16 * 16 * 16) + (x * 16 * 16) + (y * 16) + z;
                        element += embedding_matrix[i][j] * x_t[x_offset];
                        //element += embedding_matrix[i][j] * x_t[j][x][y][z];
                    }

                    if (element > largest_id_value) {
                        largest_id_value = element;
                        largest_id = i;
                    }
                }

                output_unembed[x][y][z] = largest_id;
            }
        }
    }
}

static int dummy_chunk[14][14][14] = { 6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,0,9,0,0,0,9,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,9,9,1,0,11,11,11,11,11,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,11,1,1,0,0,0,0,0,0,0,0,0,1,1,11,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,0,9,0,0,9,9,0,9,11,13,13,13,11,11,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,13,13,13,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,9,1,0,0,9,13,0,0,11,13,13,13,13,13,0,0,0,0,0,13,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,1,0,0,0,9,9,0,0,11,13,13,13,11,13,0,0,0,0,0,1,0,0,1,0,13,0,1,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,9,0,0,0,9,0,1,11,13,13,13,11,13,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,8,1,13,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,9,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,9,9,0,0,9,9,9,0,11,11,11,11,11,13,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,0,0,0,0,13,0,0,11,13,13,13,13,13,0,0,0,0,0,13,0,0,1,13,1,13,0,0,0,0,0,0,0,0,0,0,1,13,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,1,9,0,0,0,9,0,1,11,13,13,13,11,11,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,11,0,0,0,1,1,0,0,0,0,0,0,0,0,11,0,0,0,1,1,0,0,0,0,0,0,0,1,1,13,1,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,9,0,0,0,13,9,0,0,11,13,13,13,11,1,0,0,0,0,8,1,0,0,1,1,1,0,1,0,0,0,0,0,8,0,0,0,11,0,0,0,1,0,0,0,0,0,8,0,0,0,11,0,0,0,1,13,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,9,0,0,9,9,1,0,11,13,13,13,11,9,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,8,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,0,0,0,13,13,0,0,11,11,11,11,11,11,0,0,0,0,8,13,0,0,1,1,1,1,1,1,0,0,0,0,8,0,0,0,1,1,11,1,1,1,0,0,0,0,8,0,0,0,1,1,11,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,0,9,0,9,0,0,0,0,0,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,13,9,0,0,0,1,13,0,0,1,0,0,0,0,8,1,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

extern "C" __declspec(dllexport)
int32_t  __stdcall Java_net_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(void* env, void* obj, int32_t x, int32_t y, int32_t z) { // returns block_id
    //return dummy_chunk[x][y][z];
    return output_unembed[x][y][z];
}
