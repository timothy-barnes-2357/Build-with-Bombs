/**
 * @file inference_main.cpp
 * @brief This file is an interface between the Minecraft mod and the ONNX model from 
 *        PyTorch. It works by leveraging the NVIDIA TensorRT runtime to optimize and
 *        run the ONNX model. Instead of including "jni.h" for the Java Native Interface,
 *        this file simply defines functions with the correct prototype so atomic datatypes
 *        in function arguments and returns are usable from Java. 
 *
 * This program was built with TensorRT-10.5.0.18 using CUDA 12.6
 *
 * This program depends on the following NVIDIA DLLs:
 * 1. cudart64_12.dll
 * 2. nvinfer_10.dll
 * 3. nvinfer_builder_resource_10.dll
 * 4. nvonnxparser_10.dll
 * 
 */

#include <atomic>
#include <random>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <vector>

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#if defined(_MSC_VER)
#define DLL_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

/* This macro is used to print CUDA errors at a specific line number and return
 * a failed operation error code */
#define CUDA_CHECK(expression) { \
        cudaError_t err = (expression);\
        if (err != cudaSuccess) { \
            printf("CUDA error at line %d. (%s)\n", __LINE__, cudaGetErrorString(err)); \
            return INFER_ERROR_FAILED_OPERATION; \
        } \
    }

/*
 * Constants:
 */
const int VERSION_MAJOR = 0;
const int VERSION_MINOR = 1;
const int VERSION_PATCH = 0;

const int INFER_ERROR_INVALID_ARG             = 1;
const int INFER_ERROR_FAILED_OPERATION        = 2;
const int INFER_ERROR_INVALID_OPERATION       = 3;
const int INFER_ERROR_DESERIALIZE_CUDA_ENGINE = 4;
const int INFER_ERROR_BUILDING_FROM_ONNX      = 5;
const int INFER_ERROR_ENGINE_SAVE             = 6;
const int INFER_ERROR_SET_TENSOR_ADDRESS      = 7;
const int INFER_ERROR_ENQUEUE                 = 8;
const int INFER_ERROR_CREATE_RUNTIME          = 9;

const int BLOCK_ID_COUNT = 16;
const int EMBEDDING_DIMENSIONS = 3;
const int CHUNK_WIDTH = 16;

const int N_U = 5;    /* Number of inpainting steps per timestep */
const int N_T = 1000; /* Number of timesteps */

const int SIZE_X         = sizeof(float) * EMBEDDING_DIMENSIONS * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;
const int SIZE_X_CONTEXT = sizeof(float) * EMBEDDING_DIMENSIONS * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;
const int SIZE_X_MASK    = sizeof(float) *                    1 * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;

const char* onnx_file_path    = "single_update_0.1.0.onnx";
const char* engine_cache_path = "single_update_0.1.0.trt";

const float block_id_embeddings[BLOCK_ID_COUNT][EMBEDDING_DIMENSIONS] = {
    {-1.2866,  1.1751, -0.8010},
    {-1.8017, -1.0700,  1.3250},
    {-1.2621, -0.8203, -1.5511},
    {-2.8842,  0.5518,  1.2331},
    {-0.4029,  0.7936,  1.0392},
    {-1.5447, -0.2308,  1.0306},
    { 0.4594,  1.0078, -1.3730},
    { 0.1986,  0.7988, -1.2612},
    {-0.8719, -0.1313, -1.0217},
    { 0.6193, -0.0406, -1.6704},
    {-1.5885,  0.7414,  0.6155},
    { 0.0386,  2.0070,  1.4273},
    {-0.8261, -0.5659,  0.7283},
    {-0.7039,  0.5625,  0.1835},
    { 0.0673, -0.7331, -1.1119},
    { 0.0218, -0.3002,  0.3819}
};

/* 
 * State specific to each worker thread
 */
struct WorkerThreadState {
    std::mutex mtx;
    std::condition_variable cv;
    std::thread thread;
    sdd::atomic<bool> denoise_should_start;

    std::atomic<bool> diffusion_running;
    std::atomic<int32_t> timestep;

    float x_t       [EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];
    float x_t_cached[EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];
    float x_context [EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];
    float x_mask                          [CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];

    /* Middle 14^3 blocks without surrounding context */
    int cached_block_ids[CHUNK_WIDTH-2][CHUNK_WIDTH-2][CHUNK_WIDTH-2]; 
};

/* 
 * Program wide global variables and buffers:
 */
static nvinfer1::ICudaEngine* global_engine;

static std::atomic<bool> init_called;
static std::atomic<bool> init_complete;
static std::atomic<int32_t> global_last_error;
static WorkerThreadState **worker_threads;
static int worker_count;

static float alpha[N_T];
static float beta[N_T];
static float alpha_bar[N_T];

/**
 * @brief This is the main thread that's kicked off at the beginning for init.
 *        It handles the denoising process and contains all the CUDA and TensorRT code.
 *        No resources are cleaned up in this thread since it survives for the lifetime 
 *        of the program.
 *
 * @return 0 on success, error code on failure.
 */
int denoise_thread_main(WorkerThreadState *state) {

    /* 
     * The TensorRT execution context is not thread safe,
     * that's why each thread has its own context.
     */
    nvinfer1::IExecutionContext* context = global_engine->createExecutionContext();

    if (!context) {
        printf("Failed to create execution context\n");
        return INFER_ERROR_FAILED_OPERATION;
    }

    /* 
     * Allocate buffers for the inputs and outputs of the CUDA model
     * Some of these buffers are relatively large, such as the x_t buffer,
     * while others only contain a single floating point number.
     *
     * The tensor addresses must match the names on the Pytorch torch.onnx.export().
     */
    void *cuda_t, *cuda_x_t, *cuda_x_out, *cuda_x_context, *cuda_x_mask, *cuda_alpha_t, *cuda_alpha_bar_t, *cuda_beta_t;

    CUDA_CHECK(cudaMalloc(&cuda_t,           sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cuda_x_t,         SIZE_X)); // Input for each model step
    CUDA_CHECK(cudaMalloc(&cuda_x_out,       SIZE_X)); // Output produced by the model
    CUDA_CHECK(cudaMalloc(&cuda_x_context,   SIZE_X_CONTEXT));
    CUDA_CHECK(cudaMalloc(&cuda_x_mask,      SIZE_X_MASK));
    CUDA_CHECK(cudaMalloc(&cuda_alpha_t,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_alpha_bar_t, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_beta_t,      sizeof(float)));

    if (!context->setTensorAddress("t", cuda_t))                     { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("x_t", cuda_x_t))                 { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("x_out", cuda_x_out))             { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("context", cuda_x_context))       { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("mask", cuda_x_mask))             { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("alpha_t", cuda_alpha_t))         { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("alpha_bar_t", cuda_alpha_bar_t)) { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!context->setTensorAddress("beta_t", cuda_beta_t))           { return INFER_ERROR_SET_TENSOR_ADDRESS; }

    init_complete = true;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Construct the randomness generator */
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, 1.0f);
   
    /* 
     * This is the main loop. Each loop iteration represents one fully denoised chunk.
     * the start of the loop is blocked waiting on a start signal from startDiffusion()
     */
    for (;;) {

        /* Wait until the mutex unlocks */
        {
            std::unique_lock<std::mutex> lock(state->mtx);

            while (!state->denoise_should_start) {
                state->cv.wait(lock);
            }

            state->denoise_should_start = false; // Auto reset so it blocks next loop iteration.
        }

        /* Copy the "context" and "mask" tensors to the GPU */
        CUDA_CHECK(cudaMemcpy(cuda_x_context, state->x_context, SIZE_X_CONTEXT, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cuda_x_mask,    state->x_mask,    SIZE_X_MASK,    cudaMemcpyHostToDevice));

        /* Zero-out the context and mask CPU buffers so they're clean
         * for the next diffusion run. We don't need the CPU buffers anymore
         * since context and mask are already on the GPU. */
        memset(state->x_context, 0, sizeof(x_context));
        memset(state->x_mask, 0, sizeof(x_mask));
       
        /*
         * We need to fill the initial x_t with normally distributed random values.
         */
        for            (int w = 0; w < EMBEDDING_DIMENSIONS; w++) {
            for        (int x = 0; x < CHUNK_WIDTH; x++) {
               for     (int y = 0; y < CHUNK_WIDTH; y++) {
                   for (int z = 0; z < CHUNK_WIDTH; z++) {
                       state->x_t[w][x][y][z] = dist(gen);
                   }
               }
            }
        }

        /* 
         * These 'for' loops iterate over the denoising steps. The 't' steps represent the 
         * primary denoising steps whiel the 'u' steps are used to blend the known and
         * unknown regions during in-painting. 
         */
        for (int t = N_T - 1; t >= 0; t -= 1) {
            for (int u = 0; u < N_U; u++) {

                int load_index = t * N_U + u;

                /* Copy the relevant input buffers for the TensorRT model */
                CUDA_CHECK(cudaMemcpy(cuda_t, &t, sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_x_t, state->x_t, SIZE_X, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_alpha_t, &alpha[t], sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_alpha_bar_t, &alpha_bar[t], sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_beta_t, &beta[t], sizeof(float), cudaMemcpyHostToDevice));

                /* Run the model asynchronously */
                bool enqueue_succeeded = context->enqueueV3(stream);

                if (!enqueue_succeeded) {
                    printf("enqueueV3 failed\n");
                    return INFER_ERROR_ENQUEUE;
                }

                /* Block waiting for the model to complete running */
                CUDA_CHECK(cudaStreamSynchronize(stream));

                cudaError_t result;
                {
                    std::lock_guard<std::mutex> lock(state->mtx);
                    result = cudaMemcpy(state->x_t, cuda_x_out, SIZE_X, cudaMemcpyDeviceToHost);
                }

                CUDA_CHECK(result);
            }

            state->timestep = t;
            /* TODO: I should copy out the x_t only once it's completed all N_U Iterations.
             * Otherwise, I'm copying out a partially in-painted sample */
        }

        state->diffusion_running = false;
    }

    return 0; /* Never reached */
}

/** 
 * @brief This small function allows us to use the return of the denoise_thread_main
 * as the error code. This is a work-around because the C++ threading API doesn't 
 * have a way (as far as I know) to get the return value of a thread.
 */
static void denoise_thread_wrapper(WorkerThreadState *state) {
    global_last_error = denoise_thread_main(WorkerThreadState *state);
}

/** 
 * @brief Initialize the interface.
 * @return 0 on success
 */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_init(void* unused1, void* unused2,
        int worker_count) {

    if (init_called) {
        global_last_error = INFER_ERROR_INVALID_OPERATION;
        return INFER_ERROR_INVALID_OPERATION;
    }

    /* 
     * Init part 1. Create the engine so it's available to all threads 
     */

    /*
     * Read the CUDA version 
     */
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);
    printf("TensorRT version: %d\n", getInferLibVersion());
    printf("CUDA runtime version: %d\n", cuda_version);

    /* 
     * The full process for runtime is exporting is:
     *  Pytorch (torch.onnx.export()) --> ONNX (nvonnxparser) --> .TRT
     *
     * The code below first checks if we already have a TensorRT .trt file. 
     * If so, we use it. If not, we create the file by generating it from the ONNX file.
     *
     * Generating the .trt file from ONNX can take a while since TensorRT goes through a
     * long optimization process.
     */
    class Logger : public nvinfer1::ILogger { /* Logger class required by createInferRuntime()*/
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO)
                printf("%s\n", msg);
        }
    } runtime_logger;

    FILE* file = fopen(engine_cache_path, "rb");

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(runtime_logger);

    if (!runtime) {
        printf("Failed to create TensorRT runtime\n");
        return INFER_ERROR_CREATE_RUNTIME;
    }

    if (file) {
        fseek(file, 0, SEEK_END);
        size_t engine_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        std::vector<char> engine_data(engine_size);

        fread(engine_data.data(), 1, engine_size, file);
        fclose(file);

        global_engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);

        if (!global_engine) {
            printf("Failed to deserialize CUDA engine from %s\n", engine_cache_path);
            return INFER_ERROR_DESERIALIZE_CUDA_ENGINE;
        }
        printf("Loaded prebuilt TensorRT engine from %s\n", engine_cache_path);

    } else {
        /* 
         * The TensorRT .trt file wasn't found, so we need to generate it from the ONNX
         * file and cache the result for next time.
         */
        nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(runtime_logger);
        if (!builder) {
            printf("Failed to create TensorRT builder\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0);
        if (!network) {
            printf("Failed to create TensorRT network\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
        if (!config) {
            printf("Failed to create builder config\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, runtime_logger);
        if (!parser) {
            printf("Failed to create ONNX parser\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        if (!parser->parseFromFile(onnx_file_path, (int)nvinfer1::ILogger::Severity::kINFO)) {
            printf("Error parsing ONNX file: %s\n", onnx_file_path);
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }
        printf("Successfully parsed ONNX model\n");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

        nvinfer1::IHostMemory *plan = builder->buildSerializedNetwork(*network, *config);
        if (!plan) {
            printf("Failed to build serialized network\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        FILE* engine_out = fopen(engine_cache_path, "wb");

        if (!engine_out) {
            printf("Failed to save engine to %s\n", engine_cache_path);
            return INFER_ERROR_ENGINE_SAVE;
        }

        fwrite(plan->data(), 1, plan->size(), engine_out);
        fclose(engine_out);
        printf("Saved serialized engine to %s\n", engine_cache_path);
 
        global_engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
        if (!global_engine) {
            printf("Failed to deserialize CUDA engine\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }
    }

    printf("Number of layers in engine: %d\n", global_engine->getNbLayers());
    printf("Finished trt init\n");

    /* 
     * Init part 2. Calculate the denoising schedule for every timestep.
     * (These buffers are identical for all worker threads)
     *
     * This is equivalent to the Python code:
     *  beta = torch.linspace(beta1**0.5, beta2**0.5, N_T) ** 2
     *  alpha = 1 - beta
     *  alpha_bar = torch.cumprod(alpha, dim=0)
     */
    {
        float beta1 = 1e-4f;
        float beta2 = 0.02f;

        float start = sqrtf(beta1);
        float end = sqrtf(beta2);
        
        float step_size = (end - start) / (N_T - 1);

        for (int i = 0; i < N_T; i++) {

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

    /*
     * Init step 3: Create the pool of worker threads
     */
    worker_states = new WorkerThreadState*[worker_count];

    for (int i = 0; i < worker_count; i++) {

        worker_states[i] = new WorkerThreadState();
        worker_states[i].thread = std::thread(denoise_thread_wrapper, worker_states[i]);

        if (!worker_states[i].thread.joinable()) {

            printf("Thread creation failed\n");
            global_last_error = INFER_ERROR_INVALID_OPERATION;
            return INFER_ERROR_FAILED_OPERATION;
        }
    }

    init_called = true;
    return 0;
}

/**
 * @brief Check init complete
 * @return 0 on false, 1 on true
 */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getInitComplete(void* unused1, void* unused2) {

    return init_complete;
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
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_setContextBlock(void* unused1, void* unused2,
        int32_t job_id, int32_t x, int32_t y, int32_t z, int32_t block_id) {

    if (x < 0 || x >= CHUNK_WIDTH ||
        y < 0 || y >= CHUNK_WIDTH ||
        z < 0 || z >= CHUNK_WIDTH ||
        block_id < 0 || block_id >= BLOCK_ID_COUNT) {

        global_last_error = INFER_ERROR_INVALID_ARG;
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
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_startDiffusion(void* unused1, void* unused2,
        int32_t job_id) {
    
    if (diffusion_running) {
        global_last_error = INFER_ERROR_INVALID_OPERATION;
        return INFER_ERROR_INVALID_OPERATION;
    }

    global_timestep = N_T;
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
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getCurrentTimestep(void* unused1, void* unused2
        int32_t job_id) { 
    return global_timestep;
}

/** 
 * @brief cacheCurrentTimestepForReading 
 * @return Integer for cached timestep in range [0, 1000)
 * Timestep 0 is the fully denoised time.
 */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_cacheCurrentTimestepForReading(void* unused1, void* unused2,
        int32_t job_id) { 

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

/** 
 * @brief readBlockFromCachedtimestep
 * Retrieve a block_id from the cached chunk at an (x, y, z) position.
 * Integer inputs must be in range [0, 14)
 * @param: x 
 * @param: y 
 * @param: z 
 * @return: block_id of cached block.
 */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(void* unused1, void* unused2, 
        int32_t job_id, int32_t x, int32_t y, int32_t z) {

    return cached_block_ids[x][y][z];
}


/** 
 * @brief Retrieve the last error from either the diffusion thread or one of the
 *        DLL exported API functions.
 */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getLastError(void* unused1, void* unused2) {

    int32_t last_error = global_last_error;

    global_last_error = 0; // Clear the error

    return last_error;
}

/** @brief Retrieve the major version integer.*/
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getVersionMajor(void* unused1, void* unused2) {

    return (int32_t)VERSION_MAJOR;
}

/** @brief Retrieve the minor version integer */
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getVersionMinor(void* unused1, void* unused2) {

    return (int32_t)VERSION_MINOR;
}

/** @brief Retrieve the patch integer.*/
extern "C" DLL_EXPORT
int32_t Java_tbarnes_diffusionmod_Inference_getVersionPatch(void* unused1, void* unused2) {

    return (int32_t)VERSION_PATCH;
}

#if 1
/* Main function to test the interface */
int main() {

    printf("Start of main");

    /*
    int createJob();
    int setContextBlock(int job, int x, int y, int z, int block_id);
    int startDiffusion(int job);
    int getCurrentTimestep(int job);
    int cacheCurrentTimestepForReading(int job);
    int readBlockFromCachedTimestep(int job, int x, int y, int z);
    int destroyJob(int job);
    */

    int result = Java_tbarnes_diffusionmod_Inference_init(NULL, NULL);

    //Java_tbarnes_diffusionmod_Inference

    int job_id = Java_tbarnes_diffusionmod_Inference_createJob(NULL, NULL);

    if (job_id == -1) {
        printf("Failed to create job");
        return 0;
    }

    result = Java_tbarnes_diffusionmod_Inference_setContextBlock(job_id, 0, 0, 0, 0);
    
    result = Java_tbarnes_diffusionmod_Inference_startDiffusion(job_id, 0, 0);

    
    int32_t last_step = 1000;

    while (1) {

        int32_t step = Java_tbarnes_diffusionmod_Inference_getCurrentTimestep(job_id, NULL, NULL);

        if (step < last_step) {
            last_step = step;

            float sum = 0.0f;

            for (int x = 0; x < 14; x++) {
                for (int y = 0; y < 14; y++) {
                    for (int z = 0; z < 14; z++) {
                        sum += (float) Java_tbarnes_diffusionmod_Inference_readBlockFromCachedTimestep(job_id, NULL, NULL, x, y, z);

                    }
                }
            }
            
            printf("step = %d, sum = %f\n", step, sum);
            fflush(stdout);

            if (step == 0) {
                break;
            }
        }
    }

    result = Java_tbarnes_diffusionmod_Inference_cleanupJob(job_id);
    
    return 0;
}
#endif

