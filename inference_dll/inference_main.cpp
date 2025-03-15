
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include <windows.h>

#include <direct.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#define NO_ERROR 0
#define ERROR_INVALID_ARG 1
#define ERROR_FAILED_OPERATION 2
#define ERROR_INVALID_OPERATION 3

#define ID_AIR 0

#define BLOCK_ID_COUNT 96
#define EMBEDDING_DIMENSIONS 3

#define CHUNK_WIDTH 16


/* TODO:
    x- Replace asserts with actual error codes
    - Verify output is correct.
    - Create a denoising thread and move the init code into it.
    - Replace the noise bin files with cuRAND. Remove load_file().
    - Replace the denoising schedule bin files with computation in this file.
    - Replace the TRT loading with ONNX loading
    - Replace asserts with error reporting I can pass to Java
 */

using namespace nvinfer1;

static IExecutionContext* context;

CRITICAL_SECTION critical_section;

static bool init_called = false;
static bool init_complete = false;
static bool diffusion_running = false;

const int n_U = 5;    /* Number of inpainting steps per timestep */
const int n_T = 1000; /* Number of timesteps */
const int instances = n_U * n_T;

const int size_x              = 3 * 16 * 16 * 16 * sizeof(float);
const int size_x_context      = 3 * 16 * 16 * 16 * sizeof(float);
const int size_x_mask         = 1 * 16 * 16 * 16 * sizeof(float);
const int size_normal_epsilon = 3 * 16 * 16 * 16 * sizeof(float);
const int size_normal_z       = 3 * 16 * 16 * 16 * sizeof(float);
const int size_alpha     = n_T * sizeof(float);
const int size_alpha_bar = n_T * sizeof(float);
const int size_beta      = n_T * sizeof(float);

const float block_id_embeddings[BLOCK_ID_COUNT][EMBEDDING_DIMENSIONS] = {
    { 0.0, 0.0, 0.0   }, { -2.0, -1.0, 0.1 }, { 2.0, -1.0, 0.2  },  
    { 0.0, -1.0, -0.1 }, { -2.0, 2.0, -1.0 }, { -2.0, -1.0, -0.2},    
    { 0.0, -1.0, -0.3 }, { -2.0, -1.0, 0.4 }, { 2.0, 2.0, 2.0   }, 
    { 2.0, -1.0, 0.5  }, { -2.0, 2.0, 0.0  }, { 2.0, 0.0, -0.5  },  
    { 0.0, -1.0, -0.6 }, { -1.5, 1.0, 0.6  }, { 2.0, 0.0, 0.7   }, 
    { -2.0, -1.0, -0.7}, { 0.0, -1.0, 0.8  }, { 0.0, -1.0, -0.8 },   
    { 0.0, -1.0, -0.9 }, { 0.0, -1.0, 0.9  }, { 0.0, -1.0, -1.0 },   
    { 0.0, -1.0, 1.0  }, { 0.0, -1.0, 0.0  }, { -2.0, 0.0, 0.1  },  
    { 2.0, 0.0, -1.1  }, { -2.0, -1.0, -1.2}, { 0.0, -1.0, 1.1  },  
    { 0.0, -1.0, -1.3 }, { 0.0, -1.0, 1.2  }, { 0.0, -1.0, -1.4 },   
    { -2.0, 1.0, -1.5 }, { 0.5, 0.0, 0.5   }, { 0.5, 1.0, 0.5   }, 
    { 0.5, 0.0, 1.5   }, { 0.5, 1.0, 1.5   }, { 0.0, 0.5, 1.5   }, 
    { 0.0, 0.5, 0.5   }, { 1.0, 0.5, 1.5   }, { 1.0, 0.5, 0.5   }, 
    { -3.0, 1.0, -2.0 }, { -2.0, 1.0, 1.7  }, { 1.5, 1.0, -0.5  },  
    { 1.5, 2.0, -0.5  }, { 1.5, 1.0, -1.5  }, { 1.5, 2.0, -1.5  },  
    { 2.0, 1.5, -0.5  }, { 2.0, 1.5, -1.5  }, { 1.0, 1.5, -0.5  },  
    { 1.0, 1.5, -1.5  }, { 0.0, -2.0, 1.0  }, { 0.0, -1.0, 1.1  },  
    { 0.0, -1.0, -1.1 }, { 2.0, 0.0, -1.2  }, { 0.0, -1.0, 1.2  },  
    { 0.0, -1.0, -1.3 }, { 0.0, -1.0, 1.3  }, { 0.0, -1.0, -1.4 },   
    { 0.0, -1.0, 1.4  }, { 0.0, -1.0, -1.5 }, { 2.0, 0.0, 1.2   }, 
    { 2.0, 0.0, -1.6  }, { 2.0, 0.0, 1.3   }, { 2.0, 0.0, -1.7  },  
    { 2.0, 0.0, 1.4   }, { 2.0, 0.0, -1.8  }, { 2.0, 0.0, 1.5   }, 
    { 2.0, 0.0, -1.9  }, { 2.0, 0.0, 1.6   }, { 2.0, 0.0, -2.0  },  
    { 2.0, 0.0, 1.7   }, { 2.0, 0.0, -2.1  }, { 0.0, -1.0, -2.2 },   
    { 0.0, -1.0, 1.8  }, { 0.0, -1.0, -2.3 }, { 0.0, -1.0, 1.9  },  
    { 0.0, -1.0, -2.4 }, { 0.0, -1.0, 2.0  }, { 0.0, -1.0, -2.5 },   
    { 0.0, -1.0, 2.1  }, { 0.0, -1.0, -2.6 }, { 0.0, -1.0, 2.2  },  
    { 0.0, -1.0, -2.7 }, { 0.0, -1.0, 2.3  }, { 0.0, -1.0, -2.8 },   
    { 0.0, -1.0, 2.4  }, { 0.0, -1.0, -2.9 }, { 0.0, -1.0, 2.5  },  
    { 0.0, -1.0, -3.0 }, { 0.0, -1.0, 2.6  }, { 0.0, -1.0, -3.1 },   
    { 0.0, -1.0, 2.7  }, { 0.0, -1.0, -3.2 }, { 0.0, -1.0, 2.8  },  
    { 0.0, -1.0, -3.3 }, { 0.0, -1.0, 2.9  }, { 2.0, 0.0, -3.4  },  
};

static float x_t[3][16][16][16];
static float x_t_cached[3][16][16][16];
static int cached_block_ids[14][14][14]; /* Middle 14^3 blocks without surrounding context */

//static float* x_all;
//static float* x_context;
//static float* x_mask;

static float x_context[3][16][16][16];
static float x_mask[16][16][16];

static float* normal_epsilon;
static float* normal_z;
static float* alpha;
static float* alpha_bar;
static float* beta;

static void* cuda_t;
static void* cuda_x_t;
static void* cuda_x_out;
static void* cuda_x_context;
static void* cuda_x_mask;
static void* cuda_normal_epsilon;
static void* cuda_normal_z;
static void* cuda_alpha_t;
static void* cuda_alpha_bar_t;
static void* cuda_beta_t;
static void* cuda_beta_t_minus_1;
static void* cuda_post_noise_addition;

static HANDLE denoise_thread_handle;
static HANDLE start_denoise_event;

static int32_t global_timestep = 0;

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        fflush(stdout);
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

    printf("File %s contains %d bytes\n", filename, ftell_size);

    void* contents = malloc(ftell_size);

    if (!contents) {
        return NULL;
    }

    size_t bytes_read = fread(contents, 1, ftell_size, file);

    if (bytes_read != file_size) {
        printf("%s size doesn't match (%d != %d)\n", filename, bytes_read, file_size);
        exit(0);
    }

    fclose(file);

    printf("Successfully loaded %s\n", filename);
    return (float*)contents;
}


static unsigned long denoise_thread(void* unused) {

    char cwd_buffer[1024];

    if (_getcwd(cwd_buffer, sizeof(cwd_buffer)) != NULL) {
        printf("Current working directory: %s\n", cwd_buffer);
    } else {
        perror("Failed to get current working directory");
        return 1;
    }

    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);

    printf("TensorRT version: %d\n", getInferLibVersion());
    printf("CUDA runtime version: %d\n", cuda_version);

    const char* engine_file = "C:/Users/tbarnes/Desktop/projects/voxel-diffusion-minecraft-mod/model/ddim_single_update.trt";

    FILE* file = fopen(engine_file, "rb");

    if (!file) {
        printf("Error opening engine file: %s\n", engine_file);
        return ERROR_FAILED_OPERATION;
    }

    fseek(file, 0, SEEK_END);
    int engine_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* engine_data = (char*)malloc(engine_size);

    if (!engine_data) {
        return ERROR_FAILED_OPERATION;
    }

    fread(engine_data, 1, engine_size, file);
    fclose(file);

    printf("Loaded engine\n");
    
    /* Create an error logging class. This is required by the CUDA inference runtime */
    class Logger : public ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO)
                printf("%s\n", msg);
        }
    } gLogger;

    IRuntime* runtime = createInferRuntime(gLogger);

    if (!runtime) {
        return ERROR_FAILED_OPERATION;
    }

    printf("Created runtime\n\n");

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data, engine_size);
    free(engine_data);

    if (!engine) {
        return ERROR_FAILED_OPERATION;
    }

    printf("Finished deserializing CUDA engine\n\n");

    context = engine->createExecutionContext();

    if (!context) {
        return ERROR_FAILED_OPERATION;
    }

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

    printf("Number of layers in engine: %d\n", engine->getNbLayers());

    normal_epsilon = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_normal_epsilon.bin", instances * size_normal_epsilon);
    normal_z       = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_normal_z.bin", instances * size_normal_z);
    alpha          = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_alpha.bin", size_alpha);
    alpha_bar      = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_alpha_bar.bin", size_alpha_bar);
    beta           = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_beta.bin", size_beta);

    check(cudaMalloc(&cuda_t, sizeof(int32_t)));
    check(cudaMalloc(&cuda_x_t, size_x)); 
    check(cudaMalloc(&cuda_x_out, size_x)); // Output produced by the model
    check(cudaMalloc(&cuda_x_context, size_x_context));
    check(cudaMalloc(&cuda_x_mask, size_x_mask));
    check(cudaMalloc(&cuda_normal_epsilon, size_normal_epsilon));
    check(cudaMalloc(&cuda_normal_z, size_normal_z));
    check(cudaMalloc(&cuda_alpha_t, sizeof(float)));
    check(cudaMalloc(&cuda_alpha_bar_t, sizeof(float)));
    check(cudaMalloc(&cuda_beta_t, sizeof(float)));
    check(cudaMalloc(&cuda_beta_t_minus_1, sizeof(float)));
    check(cudaMalloc(&cuda_post_noise_addition, sizeof(float)));

    if (!context->setTensorAddress("t", cuda_t))                          { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("x_t", cuda_x_t))                      { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("x_out", cuda_x_out))                  { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("context", cuda_x_context))            { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("mask", cuda_x_mask))                  { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("normal_epsilon", cuda_normal_epsilon)){ return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("normal_z", cuda_normal_z))            { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("alpha_t", cuda_alpha_t))              { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("alpha_bar_t", cuda_alpha_bar_t))      { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("beta_t", cuda_beta_t))                { return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("beta_t_minus_1", cuda_beta_t_minus_1)){ return ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("post_noise_addition", cuda_post_noise_addition)){ return ERROR_FAILED_OPERATION; }

    init_complete = true;

    cudaStream_t stream;
    check(cudaStreamCreate(&stream));

    for (;;) {
        WaitForSingleObject(start_denoise_event, INFINITE);

        /* Copy the context and mask tensors to the GPU */
        check(cudaMemcpy(cuda_x_context, x_context, size_x_context, cudaMemcpyHostToDevice));
        check(cudaMemcpy(cuda_x_mask, x_mask, size_x_mask, cudaMemcpyHostToDevice));

        /* Zero-out the context and mask CPU buffers so they're clean
         * for the next diffusion run. We don't need the CPU buffers anymore
         * since context and mask are already on the GPU. */
        memset(x_context, 0, sizeof(x_context));
        memset(x_mask, 0, sizeof(x_mask));

        /* The x_t buffer needs to start with noise */
        memcpy(x_t, normal_epsilon, size_normal_epsilon);

        for (int t = n_T - 1; t >= 0; t -= 1) {
            for (int u = 0; u < n_U; u++) {
                float post_noise_addition = (u < n_U && t > 0) ? 1.0f : 0.0f;
                int load_index = t * n_U + u;

                printf("load_index %d\n", load_index);
                fflush(stdout);

                int offset_normal_epsilon = load_index * size_normal_epsilon;
                int offset_normal_z       = load_index * size_normal_z;

                float* normal_epsilon_t = (float*)((uint8_t *)normal_epsilon + offset_normal_epsilon);
                float* normal_z_t       = (float*)((uint8_t *)normal_z + offset_normal_z);

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
               
                bool enqueue_succeeded = context->enqueueV3(stream);

                if (!enqueue_succeeded) {
                    printf("enqueueV3 failed\n");
                    return 0;
                }

                check(cudaStreamSynchronize(stream));

                EnterCriticalSection(&critical_section);
                check(cudaMemcpy(x_t, cuda_x_out, size_x, cudaMemcpyDeviceToHost));
                LeaveCriticalSection(&critical_section);
            }

            global_timestep = t;
            /* TODO: I need to copy out the x_t only once it's completed all n_U iterations.
             * Otherwise, I'll be copying out a partially in-painted sample */
        }

        diffusion_running = false;
    }

    return 0;
}

/** 
 * @brief init 
 * @return 0 on success
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_init(void* unused1, void* unused2) {

    if (init_called) {
        return ERROR_INVALID_OPERATION;
    }

    /* Create the event that each run of the denoising thread is triggered by */
    start_denoise_event = CreateEvent(NULL, FALSE, FALSE, NULL);

    if (!start_denoise_event) {
        printf("CreateEvent failed (%d)\n", GetLastError());
        return ERROR_FAILED_OPERATION;
    }

    /* Create the critical section for making sure that the denoising thread isn't 
     * overwritting x_t while we're trying to cache it for reading from Java */
    InitializeCriticalSection(&critical_section);

    /* Create and launch the denoising thread */
    DWORD thread_id;
    HANDLE denoise_thread_handle = CreateThread(NULL, 0, denoise_thread, NULL, 0, &thread_id);

    if (!denoise_thread_handle) {
        printf("CreateThread error: %d\n", GetLastError());
        return ERROR_FAILED_OPERATION;
    }

    init_called = true;
    return NO_ERROR;
}

/** 
 * @brief setContextBlock 
 *  Set the context for denoising to allow the in-painting process to generate 
 *  a new chunk that matches neighbor chunks.
 * @param: x 
 * @param: y 
 * @param: z 
 * @param: block_id 
 * @return: 0 on success
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_setContextBlock(void* unused1, void* unused2,
        int32_t x, int32_t y, int32_t z, int32_t block_id) {

    if (x < 0 || x >= CHUNK_WIDTH ||
        y < 0 || y >= CHUNK_WIDTH ||
        z < 0 || z >= CHUNK_WIDTH ||
        block_id < 0 || block_id >= BLOCK_ID_COUNT) {

        return ERROR_INVALID_ARG;
    }

    /* Use the embedding matrix to find the vector for this block_id. */

    for (int dim = 0; dim < EMBEDDING_DIMENSIONS; dim++) {
        x_context[dim][x][y][z] = block_id_embeddings[block_id][dim];
    }
    
    x_mask[x][y][z] = 1.0f;

    return NO_ERROR;
}

/** 
 * @brief startDiffusion 
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_startDiffusion(void* unused1, void* unused2) {
    
    if (diffusion_running) {
        return ERROR_INVALID_OPERATION;
    }

    diffusion_running = true;
    SetEvent(start_denoise_event);

    return NO_ERROR;
}

/** 
 * @brief 
 * @return Integer for cached timestep in range [0, 1000)
 * Timestep 0 is the fully denoised time.
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_getCurrentTimestep(void* unused1, void* unused2) { 
    return global_timestep;
}

/** 
 * @brief cacheCurrentTimestepForReading 
 * @return Integer for cached timestep in range [0, 1000)
 * Timestep 0 is the fully denoised time.
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_cacheCurrentTimestepForReading(void* unused1, void* unused2) { 

    EnterCriticalSection(&critical_section);
    memcpy(x_t_cached, x_t, sizeof(x_t));
    LeaveCriticalSection(&critical_section);

    /* Perform matrix multiply of x_t and transpose(block_id_embeddings)
     * Since we only care about the index of the largest element in each row of the output
     * 4096 x BLOCK_ID_COUNT matrix, we don't need to actually store the entire matrix, 
     * just the largest element in each row. */

    for (int x = 1; x < 15; x++) {
        for (int y = 1; y < 15; y++) {
            for (int z = 1; z < 15; z++) {

                float largest_id_value = -FLT_MAX;
                int largest_id = 0;

                for (int i = 0; i < 15; i++) {
                    float element = 0.0f;

                    for (int j = 0; j < 3; j++) {

                        //int x_offset = (j * 16 * 16 * 16) + (x * 16 * 16) + (y * 16) + z;
                        //element += block_id_embeddings[i][j] * x_t[x_offset];
                        element += block_id_embeddings[i][j] * x_t_cached[j][x][y][z];
                    }

                    if (element > largest_id_value) {
                        largest_id_value = element;
                        largest_id = i;
                    }
                }

                cached_block_ids[x][y][z] = largest_id;
            }
        }
    }

    return global_timestep;
}

static int dummy_chunk[14][14][14] = { 6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,0,9,0,0,0,9,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,9,9,1,0,11,11,11,11,11,0,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,11,1,1,0,0,0,0,0,0,0,0,0,1,1,11,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,0,9,0,0,9,9,0,9,11,13,13,13,11,11,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,13,13,13,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,9,1,0,0,9,13,0,0,11,13,13,13,13,13,0,0,0,0,0,13,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,9,9,9,9,9,9,1,0,0,0,9,9,0,0,11,13,13,13,11,13,0,0,0,0,0,1,0,0,1,0,13,0,1,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,0,11,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,9,0,0,0,9,0,1,11,13,13,13,11,13,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,8,1,13,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,9,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,9,9,0,0,9,9,9,0,11,11,11,11,11,13,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,0,0,0,0,13,0,0,11,13,13,13,13,13,0,0,0,0,0,13,0,0,1,13,1,13,0,0,0,0,0,0,0,0,0,0,1,13,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,1,9,0,0,0,9,0,1,11,13,13,13,11,11,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,11,0,0,0,1,1,0,0,0,0,0,0,0,0,11,0,0,0,1,1,0,0,0,0,0,0,0,1,1,13,1,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,9,0,0,0,13,9,0,0,11,13,13,13,11,1,0,0,0,0,8,1,0,0,1,1,1,0,1,0,0,0,0,0,8,0,0,0,11,0,0,0,1,0,0,0,0,0,8,0,0,0,11,0,0,0,1,13,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,9,0,0,9,9,1,0,11,13,13,13,11,9,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,8,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,13,13,13,13,13,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,9,9,1,1,9,13,9,9,6,6,6,6,6,6,0,0,0,0,13,13,0,0,11,11,11,11,11,11,0,0,0,0,8,13,0,0,1,1,1,1,1,1,0,0,0,0,8,0,0,0,1,1,11,1,1,1,0,0,0,0,8,0,0,0,1,1,11,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,0,9,0,9,0,0,0,0,0,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,9,13,9,9,9,9,9,9,9,9,0,0,0,0,13,9,0,0,0,1,13,0,0,1,0,0,0,0,8,1,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,8,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

/** 
 * @brief readBlockFromCachedtimestep
 * Retrieve a block_id from the cached chunk at an (x, y, z) position.
 * Integer inputs must be in range [0, 14)
 * @param: x 
 * @param: y 
 * @param: z 
 * @return: block_id of cached block.
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(void* unused1, void* unused2, 
        int32_t x, int32_t y, int32_t z) {

    //return dummy_chunk[x][y][z];
    return cached_block_ids[x][y][z];
}

extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_getLastError(void* unused1, void* unused2) {

    // Check if the handle is NULL (this means that we didn't call "init" yet)
    if (denoise_thread_handle == NULL) {
        return NO_ERROR;
    }

    // "0" returns immediately without blocking. We use this to check if it's running.
    int32_t wait_result = WaitForSingleObject(denoise_thread_handle, 0);

    if (wait_result == WAIT_TIMEOUT) {
        return NO_ERROR;
    }
    
    // If we got here, the denoise_thread has completed and we can read the error code.
    unsigned long exit_code;
    GetExitCodeThread(denoise_thread_handle, &exit_code);

    return (int32_t)exit_code;
}

#if 1
void main() {

    int result = Java_tbarnes_diffusionmod_Inference_init(0, 0);
    
    result = Java_tbarnes_diffusionmod_Inference_startDiffusion(0, 0);

    printf("End of main");

    while (1) {
    }
}
#endif
