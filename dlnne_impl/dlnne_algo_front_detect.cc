#include <cuda_runtime_api.h>
#include <memory>
#include "dlnne_algo_front_detect.h"

#define CHECK(call)                                                                         \
    do{                                                                                     \
        const cudaError_t error = (call);                                                   \
        if (error != cudaSuccess) {                                                         \
            fprintf(stderr, "Error: %s:%d ", __FILE__, __LINE__);                           \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
            exit(1);                                                                        \
        }                                                                                   \
    } while(0) 

std::shared_ptr<NetworkRunner> FrontDetectorBuilder::getRunner(){
    auto runner = std::make_shared<FrontDetectorRunner>(engine_, device_id_);
    auto nb_bindings = engine_->GetNbBindings();
    auto inputDims = engine_->GetBindingDimensions(0);
    auto outputDims = engine_->GetBindingDimensions(1);

    int input_height = inputDims.d[2];
    int input_width = inputDims.d[3];

    CHECK(cudaMalloc(&(runner->d_input_), max_batch_ * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float)));
    CHECK(cudaMalloc(&(runner->d_output_), max_batch_ * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float)));
    CHECK(cudaMallocHost(&(runner->h_output_),  max_batch_ * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float)));
    runner->inputDims_ = inputDims; 
    runner->outputDims_ = outputDims;
    runner->inp_size_ = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
    runner->out_size_ = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
    return runner;
}

int FrontDetectorBuilder::get_input_size(){
    auto inputDims = engine_->GetBindingDimensions(0);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
}

void FrontDetectorRunner::infer(float* image, int batch_size){
    std::vector<void*> bindings = {d_input_, d_output_};
    timer.start();
    cudaMemcpy(d_input_, image, inp_size_, cudaMemcpyHostToDevice);
    timer.stop();
    time_htod += timer.last_elapsed();

    timer.start();
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    cudaStreamSynchronize(stream_);
    timer.stop();
    time_infer += timer.last_elapsed();

    timer.start();
    cudaMemcpy(h_output_, d_output_, out_size_, cudaMemcpyDeviceToHost);
    timer.stop();
    time_dtoh += timer.last_elapsed();

    batch_count += batch_size;
}

void FrontDetectorRunner::infer_async(float* image, int batch_size){
    std::vector<void*> bindings = {d_input_, d_output_};
    cudaMemcpyAsync(d_input_, image, inp_size_, cudaMemcpyHostToDevice, stream_);
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    cudaMemcpyAsync(h_output_, d_output_, out_size_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void FrontDetectorRunner::prerpocess(float* image, int batch_index){
    std::cout << "Not Impleted" << std::endl;
}

void FrontDetectorRunner::postprocess(float* image, int batch_index){
    std::cout << "Not Impleted" << std::endl;
}

FrontDetectorRunner::~FrontDetectorRunner(){
    if ( d_input_ != nullptr ){
        cudaFree(d_input_);
    }

    if ( d_output_ != nullptr ){
        cudaFree(d_output_);
    }

    if ( h_output_ != nullptr ){
        cudaFree(h_output_);
    }
}