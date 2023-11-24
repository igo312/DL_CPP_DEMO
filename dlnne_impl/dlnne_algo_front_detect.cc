#include <cuda_runtime_api.h>
#include <memory>
#include "dlnne_algo_front_detect.h"
#include "image_proc.h"
#define CHECK(call)                                                                         \
    do{                                                                                     \
        const cudaError_t error = (call);                                                   \
        if (error != cudaSuccess) {                                                         \
            fprintf(stderr, "Error: %s:%d ", __FILE__, __LINE__);                           \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
            exit(1);                                                                        \
        }                                                                                   \
    } while(0) 

// @todo 应该根据初始化一个类的列表，根据nb_bindings注册输入输出
// 下述的方法重写一个实例的代码量较大，但较为简单。
std::shared_ptr<NetworkRunner> FrontDetectorBuilder::getRunner(){
    auto runner = std::make_shared<FrontDetectorRunner>(engine_, device_id_);
    auto nb_bindings = engine_->GetNbBindings();
    
    auto inputDims = engine_->GetBindingDimensions(0);
    auto outputDims = engine_->GetBindingDimensions(1);
   
    int input_height = inputDims.d[2];
    int input_width = inputDims.d[3];

    // ** 实例内存的分配 **
    // 输入内存的分配
    CHECK(cudaMalloc(&(runner->d_input_beforePre_), max_batch_ * 3 * 2048 * 2048 * sizeof(uint8_t)));  // d_input_beforePre_ 假设真实输入的图片大小不会大于2048*2048*3，减少反复分配内存
    CHECK(cudaMalloc(&(runner->d_input_), max_batch_ * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float)));
    
    // 输出的总大小计算
    const int size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];
    
    // 输出内存的分配 device
    CHECK(cudaMalloc(&(runner->d_output_), max_batch_ * size * sizeof(float)));
    CHECK(cudaMalloc(&(runner->d_output_post_), max_batch_ * outputDims.d[0] * outputDims.d[2] * runner->m_max_det * sizeof(float)));

    // 输出内存的分配host
    CHECK(cudaMalloc(&(runner->h_output_), max_batch_ * size  * sizeof(float)));
    cudaMemset(runner->h_output_, 0, max_batch_ * size  * sizeof(float));

    // 输入输出大小信息的记录
    runner->inputDims_ = inputDims; 
    runner->outputDims_ = outputDims;
    runner->inp_size_ = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
    runner->inp_size_beforePre_ = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(uint8_t);
    runner->out_size_ = size;
    runner->out_size_post_ = outputDims.d[0] * runner->m_max_det * outputDims.d[2] * sizeof(float);

    return runner;
}

int FrontDetectorBuilder::get_input_size(){
    auto inputDims = engine_->GetBindingDimensions(0);
    return inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float);
}

void FrontDetectorRunner::execute(void* image, int batch_size){
    std::vector<void*> bindings = {d_input_, d_output_};
    timer.start();
    cudaMemcpyAsync(d_input_, image, inp_size_ * batch_size, cudaMemcpyHostToDevice, stream_);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_htod += timer.last_elapsed();

    timer.start();
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_infer += timer.last_elapsed();

    timer.start();
    cudaMemcpyAsync(h_output_, d_output_, out_size_ * batch_size, cudaMemcpyDeviceToHost, stream_);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_dtoh += timer.last_elapsed();

    batch_count += batch_size;
}

void FrontDetectorRunner::execute_async(void* image, int batch_size){
    std::vector<void*> bindings = {d_input_, d_output_};
    cudaMemcpyAsync(d_input_, image, inp_size_ * batch_size, cudaMemcpyHostToDevice, stream_);
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    cudaMemcpyAsync(h_output_, d_output_, out_size_ * batch_size, cudaMemcpyDeviceToHost, stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void FrontDetectorRunner::infer(void* image, int batch_size, int image_width, int image_height){
    std::vector<void*> bindings = {d_input_, d_output_};
    // 数据拷贝 htod
    timer.start();
    cudaMemcpyAsync(d_input_beforePre_, image, batch_size * 3 * image_width * image_height * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_htod += timer.last_elapsed();
    // 预处理
    timer.start();
    prerpocess(batch_size, image_width, image_height);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_pre += timer.last_elapsed();
    // 推理
    timer.start();
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_infer += timer.last_elapsed();
    // 后处理
    timer.start();
    postprocess(batch_size);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_post += timer.last_elapsed();
    // 数据拷贝 dtoh
    timer.start();
    cudaMemcpyAsync(h_output_, d_output_post_, out_size_post_ * batch_size, cudaMemcpyDeviceToHost, stream_);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_dtoh += timer.last_elapsed();
}

void FrontDetectorRunner::infer_async(void* image, int batch_size, int image_width, int image_height){
    std::vector<void*> bindings = {d_input_, d_output_};
    // 数据拷贝 htod
    cudaMemcpyAsync(d_input_beforePre_, image, batch_size * 3 * image_width * image_height * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    // 预处理
    prerpocess(batch_size, image_width, image_height);
    // 推理
    context_->Enqueue(batch_size, bindings.data(), stream_, nullptr);
    // 后处理
    postprocess(batch_size);
    // 数据拷贝 dtoh
    cudaMemcpyAsync(h_output_, d_output_post_, out_size_post_ * batch_size, cudaMemcpyDeviceToHost, stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void FrontDetectorRunner::prerpocess(int batch_size, int image_width, int image_height){
    // ** yolo 的预处理流程,下述操作做batch_size个次数**
    // 1. 获取缩放的scale
    // 2. 求解pad
    // 3. 缩放并/255

    // 一个batch的偏移量
    const int src_offsize = 3 * image_width * image_height;
    const int dst_offsize = 3 * m_input_width * m_input_height;
    for( int i = 0; i < batch_size; i++){
        // 获取当前batch的起点地址
        uint8_t* src = reinterpret_cast<uint8_t*>(d_input_beforePre_) + batch_size * src_offsize;
        float* dst = reinterpret_cast<float*>(d_input_) + batch_size * dst_offsize;
        float scale =  std::min(float(m_input_width) / image_width, float(m_input_height) / image_height);
        int img_w = scale * image_width;
        int img_h = scale * image_height;
        int pad_w = (m_input_width - img_w) / 2;
        int pad_h = (m_input_height - img_h) / 2;
        // 详情用法可参考dlpreprocess/include/image_proc.h, false代表保持不改变RGB顺序。
        RGBROIBilinearResizeNormPadPlane(src, dst, image_width, image_height, m_input_width, m_input_height, img_w, img_h, pad_w, pad_h,
                                            0,  0, image_width, image_height, scale_, mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2], pad_[0], pad_[1], pad_[2],
                                            true, stream_);
    }
}

void FrontDetectorRunner::postprocess(int batch_size){
    const int num_bboxes = outputDims_.d[0] * outputDims_.d[1];
    const int num_classes = outputDims_.d[2] - 5; // -5 是减去(x, y, w, h, conf)
    std::cout << "num bboxes output " << num_bboxes << ", num classes is " << num_classes << std::endl;

    non_max_suppression((float*)d_output_, batch_size, num_bboxes, num_classes,
                         m_conf_thres, m_iou_thres, (float*)d_output_post_,
                         m_max_det, stream_);
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

    if ( d_input_beforePre_ != nullptr ){
        cudaFree(d_input_beforePre_);
    }

    if ( d_output_post_ != nullptr ){
        cudaFree(d_output_post_);
    }
}