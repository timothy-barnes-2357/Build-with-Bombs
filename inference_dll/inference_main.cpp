/* Copyright (C) 2025 Timothy Barnes 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file inference_main.cpp
 * @brief This is an interface between the Build with Bombs Java mod and the ONNX model from 
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
 */

#include <atomic>
#include <random>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <vector>

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

/*
 * Constants:
 */
const int32_t VERSION_MAJOR = 0;
const int32_t VERSION_MINOR = 2;
const int32_t VERSION_PATCH = 0;

const int INFER_ERROR_INVALID_ARG             =  1;
const int INFER_ERROR_FAILED_OPERATION        =  2;
const int INFER_ERROR_INVALID_OPERATION       =  3;
const int INFER_ERROR_DESERIALIZE_CUDA_ENGINE =  4;
const int INFER_ERROR_BUILDING_FROM_ONNX      =  5;
const int INFER_ERROR_ENGINE_SAVE             =  6;
const int INFER_ERROR_SET_TENSOR_ADDRESS      =  7;
const int INFER_ERROR_ENQUEUE                 =  8;
const int INFER_ERROR_CREATE_RUNTIME          =  9;
const int INFER_ERROR_CUDA_RETURN             = 10;

const int BLOCK_ID_COUNT = 16;
const int EMBEDDING_DIMENSIONS = 3;
const int CHUNK_WIDTH = 16;
const int INPAINT_MARGIN = 1;

const int N_U = 5;    /* Number of inpainting steps per timestep */
const int N_T = 1000; /* Number of timesteps */

const int SIZE_X         = sizeof(float) * EMBEDDING_DIMENSIONS * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;
const int SIZE_X_CONTEXT = sizeof(float) * EMBEDDING_DIMENSIONS * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;
const int SIZE_X_MASK    = sizeof(float) *                    1 * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH;

const char* onnx_file_path    = "ddim_single_update.onnx";
const char* engine_cache_path = "ddim_single_update.trt";

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
    bool is_assigned_a_job;

    nvinfer1::IExecutionContext *context;

    std::mutex mutex;
    std::condition_variable conditional_variable;
    std::thread thread;

    std::atomic<bool> denoise_should_start;
    std::atomic<bool> diffusion_running;
    std::atomic<int32_t> timestep;

    float x_t       [EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];
    float x_context [EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];
    float x_mask                          [CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];

};

namespace { /* This is the begin of an anonymous namespace for internal linkage */

/* 
 * Program wide global variables and buffers:
 */
std::atomic<bool> init_started;
std::atomic<bool> init_complete;
std::atomic<int32_t> global_last_error;
WorkerThreadState **worker_states;
int global_worker_count;

float alpha[N_T];
float beta[N_T];
float alpha_bar[N_T];

/* Middle 14^3 blocks without surrounding context */
int cached_block_ids[CHUNK_WIDTH-2][CHUNK_WIDTH-2][CHUNK_WIDTH-2]; 

/*
 * Static functions:
 */

/** @brief Helper function+macro for printing CUDA errors */
int cuda_check(cudaError_t err, int line) {
    if (err != cudaSuccess) {

        printf("CUDA error at line %d. (%s)\n", line, cudaGetErrorString(err));

        return INFER_ERROR_CUDA_RETURN;
    }
    return 0;
}

#define CUDA_CHECK(result) { \
    int err = cuda_check((result), __LINE__); \
    if (result != 0) { \
        return err; \
    } \
}

/**
 * @brief This is the main thread that's kicked off at the beginning for init.
 *        It handles the denoising process and contains all the CUDA and TensorRT code.
 *        No resources are cleaned up in this thread since it survives for the lifetime 
 *        of the program.
 *
 * @return 0 on success, error code on failure.
 */
int thread_worker_main(WorkerThreadState *state) {

    if (init_started) {
        return INFER_ERROR_INVALID_OPERATION;
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

    if (!state->context->setTensorAddress("t", cuda_t))                     { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("x_t", cuda_x_t))                 { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("x_out", cuda_x_out))             { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("context", cuda_x_context))       { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("mask", cuda_x_mask))             { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("alpha_t", cuda_alpha_t))         { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("alpha_bar_t", cuda_alpha_bar_t)) { return INFER_ERROR_SET_TENSOR_ADDRESS; }
    if (!state->context->setTensorAddress("beta_t", cuda_beta_t))           { return INFER_ERROR_SET_TENSOR_ADDRESS; }

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
            std::unique_lock<std::mutex> lock(state->mutex);

            while (!state->denoise_should_start) {
                state->conditional_variable.wait(lock);
            }

            state->denoise_should_start = false; // Auto reset so it blocks next loop iteration.
        }

        /* Copy the "context" and "mask" tensors to the GPU */
        CUDA_CHECK(cudaMemcpy(cuda_x_mask,    state->x_mask,    SIZE_X_MASK,    cudaMemcpyHostToDevice));

        /* Zero-out the context and mask CPU buffers so they're clean
         * for the next diffusion run. We don't need the CPU buffers anymore
         * since context and mask are already on the GPU. */
        memset(state->x_context, 0, sizeof(state->x_context));
        memset(state->x_mask, 0, sizeof(state->x_mask));
       
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
                CUDA_CHECK(cudaMemcpy(cuda_x_context, state->x_context, SIZE_X_CONTEXT, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_t, &t, sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_x_t, state->x_t, SIZE_X, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_alpha_t, &alpha[t], sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_alpha_bar_t, &alpha_bar[t], sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cuda_beta_t, &beta[t], sizeof(float), cudaMemcpyHostToDevice));

                /* Run the model asynchronously */
                bool enqueue_succeeded = state->context->enqueueV3(stream);

                if (!enqueue_succeeded) {
                    printf("enqueueV3 failed\n");
                    return INFER_ERROR_ENQUEUE;
                }

                /* Block waiting for the model to complete running */
                CUDA_CHECK(cudaStreamSynchronize(stream));

                cudaError_t result;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
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
 * @brief This allows us to use the return value of the thread_worker_main/thread_init_main
 * functions as the global error code. This is a workaround because the C++ threading API
 * doesn't have an elegant way (as far as I know) to get the return value of a terminated thread.
 */
void wrapper_thread_worker_main(WorkerThreadState* state) {
    global_last_error = thread_worker_main(state);
}

/**
 * @brief This handles the initialization process. Since building a TRT engine can take a
 *        long time, we don't want to block the Java init call and instead spawn this
 *        function as a thread.
 *
 * @return 0 on success, error code on failure.
 */
int thread_init_main() {
    /* 
     * The full process for runtime export is:
     *  Pytorch (torch.onnx.export()) --> .ONNX (nvonnxparser) --> .TRT
     *
     * The code below first checks if we already have a TensorRT .trt file. 
     * If so, we use it. If not, we create the file by generating it from the ONNX file.
     *
     * Generating the .trt file from ONNX can take a while since TensorRT goes through a
     * long optimization process.
     */

    /* Allow logging printf to file */
    freopen("buildwithbombs.log", "w", stdout);

    /*
     * Read the CUDA version 
     */
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);
    printf("TensorRT version: %d\n", getInferLibVersion());
    printf("CUDA runtime version: %d\n", cuda_version);

    class Logger : public nvinfer1::ILogger { /* Logger class required by createInferRuntime()*/
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO)
                printf("%s\n", msg);
        }
    } runtime_logger;

    /* 
     * Init part 1. Create the engine so it's available to all threads 
     */
    FILE* file = fopen(engine_cache_path, "rb");

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(runtime_logger);

    if (!runtime) {
        printf("Failed to create TensorRT runtime\n");
        return INFER_ERROR_CREATE_RUNTIME;
    }

    nvinfer1::ICudaEngine *engine = nullptr;

    if (file) {
        fseek(file, 0, SEEK_END);
        size_t engine_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        std::vector<char> engine_data(engine_size);

        fread(engine_data.data(), 1, engine_size, file);
        fclose(file);

        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);

        if (!engine) {
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
 
        engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
        if (!engine) {
            printf("Failed to deserialize CUDA engine\n");
            return INFER_ERROR_BUILDING_FROM_ONNX;
        }

        /* There are a bunch of these objects we should probably destroy / free here,
         * but I still need to find an elegant way to do it */
    }

    printf("Number of layers in engine: %d\n", engine->getNbLayers());
    printf("Finished TensorRT init\n");

    /* 
     * Init part 2. Calculate the denoising schedule for every timestep.
     * (These buffers are identical for all worker threads)
     *
     * This is equivalent to the Python code:
     *  beta = torch.linspace(beta1**0.5, beta2**0.5, N_T) ** 2
     *  alpha = 1 - beta
     *  alpha_bar = torch.cumprod(alpha, dim=0)
     *
     * TODO: These buffers could be computed compile-time with a constexpr.
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
    worker_states = new WorkerThreadState*[global_worker_count];

    for (int i = 0; i < global_worker_count; i++) {

        worker_states[i] = new WorkerThreadState();

        /* 
         * The TensorRT execution context is not thread safe,
         * that's why each thread has its own context.
         */
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();

        if (!context) {
            printf("Failed to create execution context\n");
            return INFER_ERROR_FAILED_OPERATION;
        }

        worker_states[i]->context = context;

        std::thread worker = std::thread(wrapper_thread_worker_main, worker_states[i]);

        worker.detach(); /* Allow the thread to run beyond the lifetime of this scope */

        printf("Created worker %d\n", i);
    }

    init_complete = true;

    return 0;
}

void wrapper_thread_init() {
    global_last_error = thread_init_main();
}

/*
 * Exported DLL functions:
 */

/** 
 * @brief Initialize TensorRT and all the worker threads.
 * @return 0 on success
 */
int32_t startInit(int worker_count) {

    int error_code = 0;

    if (init_started) {
        error_code = INFER_ERROR_INVALID_OPERATION;
    } else {

        global_worker_count = worker_count; 
        std::thread init_thread = std::thread(thread_init_main);

        init_thread.detach();
    }

    if (error_code != 0) {
        global_last_error = error_code;
    }

    return error_code;
}

/**
 * @brief Check init complete
 * @return 0 if not complete, 1 if init is complete 
 */
int32_t getInitComplete() {

    return init_complete;
}

/** @brief Create a new job drawn from the existing thread pool.
 *  @return job_id on success (>= 0), -1 on failure to create job
 */
int32_t createJob() {

    int job_id = -1;

    if (init_complete) {
        for (int i = 0; i < global_worker_count; i++) {

            if (!worker_states[i]->is_assigned_a_job) {

                worker_states[i]->is_assigned_a_job = true;
                job_id = i;

                break;
            }
        }
    }

    return job_id;
}

/** @brief Destroy a job to free the worker thread.
 *  @return 0 on success, invalid_operation on failure.
 */
int32_t destroyJob(int32_t job_id) {

    int32_t error_code = INFER_ERROR_INVALID_OPERATION;

    if (init_complete) {
        if (job_id >= 0 && job_id < global_worker_count) { 

            WorkerThreadState *state = worker_states[job_id];

            if (!state->diffusion_running) {

                if (state->is_assigned_a_job) {
                    state->is_assigned_a_job = false;
                    error_code = 0;
                }
            }
        }
    }

    if (error_code != 0) {
        global_last_error = error_code;
    }

    return error_code;
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
int32_t setContextBlock(int32_t job_id, int32_t x, int32_t y, int32_t z, int32_t block_id) {

    if (!init_complete) {
        global_last_error = INFER_ERROR_INVALID_OPERATION;
        return global_last_error;
    }

    if (x < 0 || x >= CHUNK_WIDTH ||
        y < 0 || y >= CHUNK_WIDTH ||
        z < 0 || z >= CHUNK_WIDTH ||
        block_id < 0 || block_id >= BLOCK_ID_COUNT ||
        job_id < 0 || job_id >= global_worker_count) {

        global_last_error = INFER_ERROR_INVALID_ARG;
        return global_last_error;
    }

    WorkerThreadState *state = worker_states[job_id];

    /* Use the embedding matrix to find the vector for this block_id. */
    for (int dim = 0; dim < EMBEDDING_DIMENSIONS; dim++) {
        state->x_context[dim][x][y][z] = block_id_embeddings[block_id][dim];
    }

    state->x_mask[x][y][z] = 1.0f;

    return 0;
}

/** 
 * @brief startDiffusion 
 * @return 0 on success, invalid_operation on failure
 */
int32_t startDiffusion(int32_t job_id) {

    int error_code = INFER_ERROR_INVALID_OPERATION;

    if (init_complete && job_id >= 0 && job_id < global_worker_count) {

        WorkerThreadState *state = worker_states[job_id];

        /* Check if diffusion is already running */
        if (!state->diffusion_running) {

            state->timestep = N_T;
            state->diffusion_running = true;

            std::lock_guard<std::mutex> lock(state->mutex);
            state->denoise_should_start = true;
            state->conditional_variable.notify_one();

            error_code = 0;
        }
    }

    if (error_code != 0) {
        global_last_error = error_code;
    }

    return error_code;
}

/** 
 * @brief 
 * @return Integer for cached timestep in range [0, 1000). Returns -1 on failure.
 * Timestep 0 is the fully denoised time.
 */
int32_t getCurrentTimestep(int32_t job_id) { 

    int result = -1;

    if (init_complete && job_id >= 0 && job_id < global_worker_count) {
        result = worker_states[job_id]->timestep;
    }

    return result;
}

/** 
 * @brief cacheCurrentTimestepForReading 
 * @return Integer for cached timestep in range [0, 1000). Returns -1 on Failure.
 * Timestep 0 is the fully denoised time.
 */
int32_t cacheCurrentTimestepForReading(int32_t job_id) { 

    /* This is a temporary buffer to store the worker's x_t buffer so we don't block
     * that worker while we're performing the unembedding process below. */
    static float timestep_cache[EMBEDDING_DIMENSIONS][CHUNK_WIDTH][CHUNK_WIDTH][CHUNK_WIDTH];

    if (!init_complete || job_id < 0 || job_id >= global_worker_count) {
        return -1;
    }

    WorkerThreadState *state = worker_states[job_id];
    
    /* Lock before we copy x_t to the cache */
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        memcpy(timestep_cache, state->x_t, sizeof(state->x_t));
    }

    /* Perform matrix multiply of x_t and transpose(block_id_embeddings)
     * Since we only care about the index of the smallest element in each row of the output
     * 4096 x BLOCK_ID_COUNT matrix, we don't need to actually store the entire matrix. */

    const int start = INPAINT_MARGIN;
    const int end = CHUNK_WIDTH - INPAINT_MARGIN;

    for (int x = start; x < end; x++) {
        for (int y = start; y < end; y++) {
            for (int z = start; z < end; z++) {

                float min_distance = FLT_MAX;
                int closest_id = 0;

                for (int i = 0; i < BLOCK_ID_COUNT; i++) {
                    float distance = 0.0f;

                    for (int j = 0; j < EMBEDDING_DIMENSIONS; j++) {
                        float diff = timestep_cache[j][x][y][z] - block_id_embeddings[i][j];
                        distance += diff * diff;
                    }

                    if (distance < min_distance) {
                        min_distance = distance;
                        closest_id = i;
                    }
                }

                cached_block_ids[x-INPAINT_MARGIN][y-INPAINT_MARGIN][z-INPAINT_MARGIN] = closest_id;
            }
        }
    }

    return state->timestep;
}

/** 
 * @brief readBlockFromCachedtimestep
 * Retrieve a block_id from the cached chunk at an (x, y, z) position.
 * Integer inputs must be in range [0, 14)
 * @param: x 
 * @param: y 
 * @param: z 
 * @return: block_id of cached block, -1 on failure.
 */
int32_t readBlockFromCachedTimestep(int32_t x, int32_t y, int32_t z) {

    return cached_block_ids[x][y][z];
}

/** 
 * @brief Retrieve the last error from either the diffusion thread or one of the
 *        DLL exported API functions.
 */
int32_t getLastError() {

    int32_t last_error = global_last_error;

    global_last_error = 0; // Clear the error

    return last_error;
}


} /* This is the end of the internal linkage namespace. Functions below this are
   * exported from the DLL*/

/* 
 * JNI Export:
 *
 * The code below exports the DLL functions with the correct name and arguments
 * for the Java Native Interface to use. Each function start with the full path and
 * name of the Java module (Java_com_buildwithbombs_Inference) and also take pointers
 * as the first two arguments (JNIEnv *, jobject), but we're not using either pointer
 * since integer input to the function and integer return suffices for us. This has the
 * added benefit that we don't need to build headers with Java, we can just directly use
 * these exports. 
 *
 * On Windows, the JNI requires the __stdcall calling convention. This requires using 
 * the "/Gz" flag in MSVC. Linux doesn't require this since it's the default CDECL.
 */
#if defined(_MSC_VER)
#define DLL_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define DLL_EXPORT __attribute__((visibility("default")))
#endif

extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_startInit(void* unused1, void* unused2, int32_t worker_count) { return startInit(worker_count); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getInitComplete(void* unused1, void* unused2) { return getInitComplete(); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_createJob(void* unused1, void* unused2) { return createJob(); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_destroyJob(void* unused1, void* unused2, int32_t job_id) { return destroyJob(job_id); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_setContextBlock(void* unused1, void* unused2, int32_t job_id, int32_t x, int32_t y, int32_t z, int32_t block_id) { return setContextBlock(job_id, x, y, z, block_id); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_startDiffusion(void* unused1, void* unused2, int32_t job_id) { return startDiffusion(job_id); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getCurrentTimestep(void* unused1, void* unused2, int32_t job_id) { return getCurrentTimestep(job_id); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_cacheCurrentTimestepForReading(void* unused1, void* unused2, int32_t job_id) { return cacheCurrentTimestepForReading(job_id); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_readBlockFromCachedTimestep(void* unused1, void* unused2, int32_t x, int32_t y, int32_t z) { return readBlockFromCachedTimestep(x, y, z); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getLastError(void* unused1, void* unused2) { return getLastError(); }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getVersionMajor(void* unused1, void* unused2) { return VERSION_MAJOR; }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getVersionMinor(void* unused1, void* unused2) { return VERSION_MINOR; }
extern "C" DLL_EXPORT int32_t Java_com_buildwithbombs_Inference_getVersionPatch(void* unused1, void* unused2) { return VERSION_PATCH; }

/*
 * Finally, below is some brief code I use to test the above functions.
 * It creates multiple diffusion jobs running simultaneously.
 * Uncomment if building as a .exe and not .dll
 */
#if 0
int main() {

    /* In the nomenclature of this file, "workers" are persistent threads that
     * take on "jobs" which are tasks to fully diffuse a chunk). There can be more
     * workers than jobs, but there can never be more jobs than workers 
     */

    const int job_count = 2; /* How many active jobs we're creating */
    int32_t jobs[job_count]; /* ID for reach job we create */
    int32_t last_job_step[job_count]; /* timestep for each job */

    if(startInit(job_count) != 0) {
        printf("startInit failed\n");
    } else {
        printf("Init started\n");
    }

    /* Wait until initialization is complete.
     * This can take awhile if we need to build a new .trt engine */
    for (;;) {
        int32_t init_complete = getInitComplete();

        if (init_complete == 1) {
            printf("Init complete\n");
            break;
        }
    }

    /* Create all our jobs */
    for (int i = 0; i < job_count; i++) {
        jobs[i] = createJob();
    }

    /* The first job will contain all dirt in the context.
     * The other jobs contexts default to all air*/
    for (int x = 0; x < CHUNK_WIDTH; x++) {
        for (int y = 0; y < CHUNK_WIDTH; y++) {
            for (int z = 0; z < CHUNK_WIDTH; z++) {

                const int dirt_id = 1;

                if (setContextBlock(jobs[0], x, y, z, dirt_id) != 0) {
                    printf("setContextBlock failed\n");
                }
            }
        }
    }

    for (int i = 0; i < job_count; i++) {
        last_job_step[i] = 1000;

        if (startDiffusion(jobs[i]) != 0) {
            printf("startDiffusion (for job %d) failed\n", i);
        }
    }
   
    /* Collect and print all 1k timesteps for all active jobs */
    for (;;) {

        int jobs_finished = 0;

        for (int i = 0; i < job_count; i++) {

            int32_t step = getCurrentTimestep(jobs[i]);

            if (step < last_job_step[i]) {
                last_job_step[i] = step;

                if(cacheCurrentTimestepForReading(jobs[i]) == -1) {
                    printf("cacheCurrentTimestep failed\n");
                }

                int32_t sum = 0.0f;
                const int width = CHUNK_WIDTH - (INPAINT_MARGIN * 2);

                for (int x = 0; x < width; x++) {
                    for (int y = 0; y < width; y++) {
                        for (int z = 0; z < width; z++) {
                            sum += readBlockFromCachedTimestep(x, y, z);
                        }
                    }
                }
                
                printf("job: %d, step = %d, sum = %f\n", i, last_job_step[i], sum);
                fflush(stdout);

                if (last_job_step[i] == 0) {
                    jobs_finished++;
                }
            }
        }

        if (jobs_finished == job_count) {
            break;
        }
    }

    for (int i = 0; i < job_count; i++) {

        if(destroyJob(jobs[i]) != 0) {
            printf("Destroy job failed\n");
        }
    }
    
    return 0;
}
#endif

