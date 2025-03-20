#include <random>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#define INFER_ERROR_INVALID_ARG 1
#define INFER_ERROR_FAILED_OPERATION 2
#define INFER_ERROR_INVALID_OPERATION 3

#define ID_AIR 0

#define BLOCK_ID_COUNT 96
#define EMBEDDING_DIMENSIONS 3

#define CHUNK_WIDTH 16

static std::mutex mtx;
static std::condition_variable cv;
static bool denoise_should_start;
static std::thread denoise_thread;

/* TODO:
 * - Try moving away from prebuilt TRT runtime.
 * - Make better error reporting and present in Minecraft.
 * - Use a cross platform threading API.
 * - Remove dead code and add substantial comments.
 */

using namespace nvinfer1;

static IExecutionContext* context;

//CRITICAL_SECTION critical_section;

static volatile bool init_called = false;
static volatile bool init_complete = false;
static volatile bool diffusion_running = false;

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

static float* x_initial;
static float* x_all;
//static float* x_context;
//static float* x_mask;

static float x_context[3][16][16][16];
static float x_mask[16][16][16];

//static float* normal_epsilon;
//static float* normal_z;
//static float* alpha;
//static float* alpha_bar;
//static float* beta;

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

static volatile int32_t global_timestep = 0;

#define cuda_check(expression) { \
        cudaError_t err = (expression);\
        if (err != cudaSuccess) { \
            printf("CUDA error at line %d. (%s)\n", __LINE__, cudaGetErrorString(err)); \
            return INFER_ERROR_FAILED_OPERATION; \
        } \
    }

#define curand_check(expression) { \
        curandStatus_t err = (expression); \
        if (err != CURAND_STATUS_SUCCESS) { \
            printf("cuRAND Error: %d\n", err); \
            return INFER_ERROR_FAILED_OPERATION; \
        } \
    }

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA: %s\n", cudaGetErrorString(err));
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
        printf("%s size doesn't match (%u != %u)\n", filename, bytes_read, file_size);
        exit(0);
    }

    fclose(file);

    printf("Successfully loaded %s\n", filename);
    return (float*)contents;
}


static float alpha[n_T];
static float beta[n_T];
static float alpha_bar[n_T];

int denoise_thread_main() {

    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);

    printf("TensorRT version: %d\n", getInferLibVersion());
    printf("CUDA runtime version: %d\n", cuda_version);

    const char* engine_file = "C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/ddim_single_update.trt";

    FILE* file = fopen(engine_file, "rb");

    if (!file) {
        printf("Error opening engine file: %s\n", engine_file);
        return INFER_ERROR_FAILED_OPERATION;
    }

    fseek(file, 0, SEEK_END);
    int engine_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* engine_data = (char*)malloc(engine_size);

    if (!engine_data) {
        return INFER_ERROR_FAILED_OPERATION;
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
    } runtime_logger;

    IRuntime* runtime = createInferRuntime(runtime_logger);

    if (!runtime) {
        return INFER_ERROR_FAILED_OPERATION;
    }

    printf("Created runtime\n\n");

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data, engine_size);
    free(engine_data);

    if (!engine) {
        return INFER_ERROR_FAILED_OPERATION;
    }

    printf("Finished deserializing CUDA engine\n\n");

    context = engine->createExecutionContext();

    if (!context) {
        return INFER_ERROR_FAILED_OPERATION;
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

            if (j < dims.nbDims - 1) {
                printf(", ");
            }
        }
        printf("], Size in Bytes = %d\n", totalSize);
    }

    printf("Number of layers in engine: %d\n", engine->getNbLayers());

    x_initial = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_x_initial.bin", size_x);
    x_all     = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_x_all.bin", instances * size_x);

    //normal_epsilon = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_normal_epsilon.bin", instances * size_normal_epsilon);
    //normal_z       = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_normal_z.bin", instances * size_normal_z);
    //alpha          = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_alpha.bin", size_alpha);
    //alpha_bar      = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_alpha_bar.bin", size_alpha_bar);
    //beta           = load_file("C:/Users/tbarnes/Desktop/projects/voxelnet/experiments/TestTensorRT/save_beta.bin", size_beta);

    float beta1 = 1e-4f;
    float beta2 = 0.02f;

    { /* Compute the denoising schedule for ever timestep*/
        float start = sqrtf(beta1);
        float end = sqrtf(beta2);
        
        float step_size = (end - start) / (n_T - 1);

        for (int i = 0; i < n_T; i++) {

            float result = start + step_size*i;
            beta[i] = result * result;

            alpha[i] = (1.0f - beta[i]);

            // Alpha bar is the cumulative product.
            alpha_bar[i] = alpha[i];

            if (i > 0) {
                alpha_bar[i] =  alpha_bar[i] * alpha_bar[i-1];
            }
        }
    }

    cuda_check(cudaMalloc(&cuda_t, sizeof(int32_t)));
    cuda_check(cudaMalloc(&cuda_x_t, size_x)); 
    cuda_check(cudaMalloc(&cuda_x_out, size_x)); // Output produced by the model
    cuda_check(cudaMalloc(&cuda_x_context, size_x_context));
    cuda_check(cudaMalloc(&cuda_x_mask, size_x_mask));
    //cuda_check(cudaMalloc(&cuda_normal_epsilon, size_normal_epsilon));
    //cuda_check(cudaMalloc(&cuda_normal_z, size_normal_z));
    cuda_check(cudaMalloc(&cuda_alpha_t, sizeof(float)));
    cuda_check(cudaMalloc(&cuda_alpha_bar_t, sizeof(float)));
    cuda_check(cudaMalloc(&cuda_beta_t, sizeof(float)));
    //cuda_check(cudaMalloc(&cuda_beta_t_minus_1, sizeof(float)));
    //cuda_check(cudaMalloc(&cuda_post_noise_addition, sizeof(float)));

    if (!context->setTensorAddress("t", cuda_t))                          { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("x_t", cuda_x_t))                      { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("x_out", cuda_x_out))                  { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("context", cuda_x_context))            { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("mask", cuda_x_mask))                  { return INFER_ERROR_FAILED_OPERATION; }
    //if (!context->setTensorAddress("normal_epsilon", cuda_normal_epsilon)){ return INFER_ERROR_FAILED_OPERATION; }
    //if (!context->setTensorAddress("normal_z", cuda_normal_z))            { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("alpha_t", cuda_alpha_t))              { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("alpha_bar_t", cuda_alpha_bar_t))      { return INFER_ERROR_FAILED_OPERATION; }
    if (!context->setTensorAddress("beta_t", cuda_beta_t))                { return INFER_ERROR_FAILED_OPERATION; }
    //if (!context->setTensorAddress("beta_t_minus_1", cuda_beta_t_minus_1)){ return INFER_ERROR_FAILED_OPERATION; }
    //if (!context->setTensorAddress("post_noise_addition", cuda_post_noise_addition)){ return INFER_ERROR_FAILED_OPERATION; }

    init_complete = true;

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));
    
    for (;;) {

        /* Wait until the mutex unlocks */
        {
            std::unique_lock<std::mutex> lock(mtx);

            while (!denoise_should_start) {
                cv.wait(lock);
            }

            denoise_should_start = false; // Auto reset so it blocks next loop iteration.
        }

        /* Fill in the middle 14^3 voxels of the mask*/
        for         (int x = 1; x < 15; x++) {
            for     (int y = 1; y < 15; y++) {
                for (int z = 1; z < 15; z++) {
                    x_mask[x][y][z] = 1.0f;
                }
            }
        }

        /* Copy the "context" and "mask" tensors to the GPU */
        cuda_check(cudaMemcpy(cuda_x_context, x_context, size_x_context, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(cuda_x_mask, x_mask, size_x_mask, cudaMemcpyHostToDevice));

        /* Zero-out the context and mask CPU buffers so they're clean
         * for the next diffusion run. We don't need the CPU buffers anymore
         * since context and mask are already on the GPU. */
        //memset(x_context, 0, sizeof(x_context));
        //memset(x_mask, 0, sizeof(x_mask));
        //
        
        //float smallarray[5];
        //curandGenerator_t rand_generator;
        //curand_check(curandCreateGeneratorHost(&rand_generator, CURAND_RNG_PSEUDO_XORWOW));
        //curand_check(curandSetPseudoRandomGeneratorSeed(rand_generator, 42));
        //curand_check(curandGenerateNormal(rand_generator, smallarray, 5, 0.0f, 1.0f));

        std::random_device rd;  // Seed generator
        std::mt19937 gen(rd()); // Mersenne Twister engine
        std::normal_distribution<float> dist(0.0f, 1.0f);
                                                         
        float *arr = &x_t[0][0][0][0];

        for (size_t i = 0; i < size_x/sizeof(float); ++i) {
            //arr[i] = dist(gen);
        }

        /* The x_t buffer needs to start with noise */
        //memcpy(x_t, x_initial, size_x);
        bool first_iteration_complete = false;

        for (int t = n_T - 1; t >= 0; t -= 1) {
            for (int u = 0; u < n_U; u++) {

                float post_noise_addition = (u < n_U && t > 0) ? 1.0f : 0.0f;
                int load_index = t * n_U + u;

                //printf("load_index %d\n", load_index);
                //fflush(stdout);

                int offset_normal_epsilon = load_index * size_normal_epsilon;
                int offset_normal_z       = load_index * size_normal_z;

                //float* normal_epsilon_t = (float*)((uint8_t *)normal_epsilon + offset_normal_epsilon);
                //float* normal_z_t       = (float*)((uint8_t *)normal_z + offset_normal_z);

                //float alpha_t = alpha[t];
                //float alpha_bar_t = alpha_bar[t];
                //float beta_t = beta[t];
                //float beta_t_minus_1 = t > 0 ? beta[t - 1] : 0.0;

                cuda_check(cudaMemcpy(cuda_t, &t, sizeof(int32_t), cudaMemcpyHostToDevice));

                //if (!first_iteration_complete) {
                //    curand_check(curandGenerateNormal(rand_generator, (float *)cuda_x_t, size_x, 0.0f, 1.0f));
                //} else {
                    cuda_check(cudaMemcpy(cuda_x_t, x_t, size_x, cudaMemcpyHostToDevice));
                //}
                //cuda_check(cudaMemcpy(cuda_normal_epsilon, normal_epsilon_t, size_normal_epsilon, cudaMemcpyHostToDevice));
                //cuda_check(cudaMemcpy(cuda_normal_z, normal_z_t, size_normal_z, cudaMemcpyHostToDevice));

                cuda_check(cudaMemcpy(cuda_alpha_t, &alpha[t], sizeof(float), cudaMemcpyHostToDevice));
                cuda_check(cudaMemcpy(cuda_alpha_bar_t, &alpha_bar[t], sizeof(float), cudaMemcpyHostToDevice));
                cuda_check(cudaMemcpy(cuda_beta_t, &beta[t], sizeof(float), cudaMemcpyHostToDevice));
                //cuda_check(cudaMemcpy(cuda_beta_t_minus_1, &beta_t_minus_1, sizeof(float), cudaMemcpyHostToDevice));
                //cuda_check(cudaMemcpy(cuda_post_noise_addition, &post_noise_addition, sizeof(float), cudaMemcpyHostToDevice));

                bool enqueue_succeeded = context->enqueueV3(stream);

                if (!enqueue_succeeded) {
                    printf("enqueueV3 failed\n");
                    return 0;
                }

                cuda_check(cudaStreamSynchronize(stream));

                cudaError_t result;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    result = cudaMemcpy(x_t, cuda_x_out, size_x, cudaMemcpyDeviceToHost);
                }

                //EnterCriticalSection(&critical_section);
                //LeaveCriticalSection(&critical_section);

                cuda_check(result);

#if 0
                float *x_base = &x_t[0][0][0][0];
                int x_count = size_x/sizeof(float);

                for (int i = 0; i < 8; i++) {
                    printf("%f, ", x_base[i]);
                }

                printf("\n");

                /* Remove this later */
                memcpy(x_t_cached, x_t, sizeof(x_t));

                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        for (int z = 0; z < 16; z++) {

                            float min_distance = FLT_MAX;
                            int closest_id = 0;

                            for (int i = 0; i < BLOCK_ID_COUNT; i++) {
                                float distance = 0.0f;

                                for (int j = 0; j < EMBEDDING_DIMENSIONS; j++) {
                                    float a = x_t_cached[j][x][y][z];
                                    float b = block_id_embeddings[i][j];
                                    float diff = a - b;
                                    distance += diff * diff;
                                }

                                distance = sqrtf(distance);

                                printf("* %f \n", distance);

                                if (distance < min_distance) {
                                    min_distance = distance;
                                    closest_id = i;
                                }
                            }
                            printf("Closest id %d\n", closest_id);
                            //cached_block_ids[x-1][y-1][z-1] = closest_id;
                        }
                    }
                }

                for (int i = 0; i < 8; i++) {
                    printf("%d, ", cached_block_ids[0][0][i]);
                }

                printf("\n");

                if (t < 998) {
                    return 0;
                }
                //return 0;
#endif
            }

            global_timestep = t;
            /* TODO: I should copy out the x_t only once it's completed all n_U iterations.
             * Otherwise, I'll be copying out a partially in-painted sample */

            first_iteration_complete = true;
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
        return INFER_ERROR_INVALID_OPERATION;
    }

    ///* Create the event that each run of the denoising thread is triggered by */
    //start_denoise_event = CreateEvent(0, 0, 0, 0);

    //if (!start_denoise_event) {
    //    printf("CreateEvent failed (%d)\n", GetLastError());
    //    return INFER_ERROR_FAILED_OPERATION;
    //}

    ///* Create the critical section for making sure that the denoising thread isn't 
    // * overwritting x_t while we're trying to cache it for reading from Java */
    //InitializeCriticalSection(&critical_section);

    ///* Create and launch the denoising thread */
    //DWORD thread_id;
    //HANDLE denoise_thread_handle = CreateThread(0, 0, denoise_thread, 0, 0, &thread_id);

    //if (!denoise_thread_handle) {
    //    printf("CreateThread error: %d\n", GetLastError());
    //    return INFER_ERROR_FAILED_OPERATION;
    //}

    denoise_thread = std::thread(denoise_thread_main);

    if (!denoise_thread.joinable()) {
        printf("Thread creation failed\n");
        return INFER_ERROR_FAILED_OPERATION;
    }

    init_called = true;
    return 0;
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

        return INFER_ERROR_INVALID_ARG;
    }

    /* Use the embedding matrix to find the vector for this block_id. */

    for (int dim = 0; dim < EMBEDDING_DIMENSIONS; dim++) {
        x_context[dim][x][y][z] = block_id_embeddings[block_id][dim];
    }
    
    x_mask[x][y][z] = 1.0f;

    return 0;
}

/** 
 * @brief startDiffusion 
 */
extern "C" __declspec(dllexport)
int32_t Java_tbarnes_diffusionmod_Inference_startDiffusion(void* unused1, void* unused2) {
    
    if (diffusion_running) {
        return INFER_ERROR_INVALID_OPERATION;
    }

    global_timestep = n_T;
    diffusion_running = true;

    {
        std::lock_guard<std::mutex> lock(mtx);
        denoise_should_start = true;
        cv.notify_one();
    }

    return 0;
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

    {
        std::lock_guard<std::mutex> lock(mtx);
        memcpy(x_t_cached, x_t, sizeof(x_t));
    }

    /* Perform matrix multiply of x_t and transpose(block_id_embeddings)
     * Since we only care about the index of the smallest element in each row of the output
     * 4096 x BLOCK_ID_COUNT matrix, we don't need to actually store the entire matrix. */

    for (int x = 1; x < 15; x++) {
        for (int y = 1; y < 15; y++) {
            for (int z = 1; z < 15; z++) {

                float min_distance = FLT_MAX;
                int closest_id = 0;

                for (int i = 0; i < BLOCK_ID_COUNT; i++) {
                    float distance = 0.0f;

                    for (int j = 0; j < EMBEDDING_DIMENSIONS; j++) {
                        float diff = x_t_cached[j][x][y][z] - block_id_embeddings[i][j];
                        distance += diff * diff;
                    }

                    if (distance < min_distance) {
                        min_distance = distance;
                        closest_id = i;
                    }
                }

                cached_block_ids[x-1][y-1][z-1] = closest_id;
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

//    // Check if the handle is NULL (this means that we didn't call "init" yet)
//    if (denoise_thread_handle == NULL) {
//        return NO_ERROR;
//    }
//
//    // "0" returns immediately without blocking. We use this to check if it's running.
//    int32_t wait_result = WaitForSingleObject(denoise_thread_handle, 0);
//
//    if (wait_result == WAIT_TIMEOUT) {
//        return NO_ERROR;
//    }
//    
//    // If we got here, the denoise_thread has completed and we can read the error code.
//    unsigned long exit_code;
//    GetExitCodeThread(denoise_thread_handle, &exit_code);

    int32_t exit_code = 0;
    return (int32_t)exit_code;
}

#if 0
void main() {

    int result = Java_tbarnes_diffusionmod_Inference_init(0, 0);
    
    result = Java_tbarnes_diffusionmod_Inference_startDiffusion(0, 0);

    printf("End of main");

    int32_t last_step = 1000;

    while (1) {

        int32_t step = Java_tbarnes_diffusionmod_Inference_getCurrentTimestep(NULL, NULL);

        if (step < last_step) {
            last_step = step;


            //Java_tbarnes_diffusionmod_Inference_cacheCurrentTimestepForReading(NULL, NULL);

            float sum = 0.0f;

            for (int x = 0; x < 14; x++) {
                for (int y = 0; y < 14; y++) {
                    for (int z = 0; z < 14; z++) {
                        //int32_t block_id = Java_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(NULL, NULL, 
                        //        x, y, z);
                        //if (block_id != 0) {
                        //    non_air_count += 1;
                        //}
                        //
                        //sum += x_t[0][x][y][z] + x_t[1][x][y][z] + x_t[2][x][y][z];
                        sum += (float) Java_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(NULL, NULL, x, y, z);

                    }
                }
            }
            
            printf("step = %d, sum = %f\n", step, sum);
            fflush(stdout);
        }
    }
}
#endif

