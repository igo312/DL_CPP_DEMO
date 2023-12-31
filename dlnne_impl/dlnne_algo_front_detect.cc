#include <cuda_runtime_api.h>
#include <memory>
#include <cstring>
#include "dlnne_algo_front_detect.h"
#include "image_proc.h"
#include <fstream>
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
    std::cout << "get runner max_batch: " << max_batch_ << std::endl;
    CHECK(cudaMalloc(&(runner->d_input_beforePre_), max_batch_ * 3 * 2048 * 2048 * sizeof(uint8_t)));  // d_input_beforePre_ 假设真实输入的图片大小不会大于2048*2048*3，减少反复分配内存
    CHECK(cudaMalloc(&(runner->d_input_), max_batch_ * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float)));
    
    // 输出的总大小计算
    const int size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];
    
    // 输出内存的分配 device
    CHECK(cudaMalloc(&(runner->d_output_), max_batch_ * size * sizeof(float)));
    CHECK(cudaMalloc(&(runner->d_output_post_), max_batch_ * outputDims.d[0] * outputDims.d[2] * runner->m_max_det * sizeof(float)));

    // 输出内存的分配host
    CHECK(cudaMallocHost(&(runner->h_output_), max_batch_ * size  * sizeof(float)));

    // 初始化值
    CHECK(cudaMemset(runner->d_output_post_, 0,  max_batch_ * outputDims.d[0] * outputDims.d[2] * runner->m_max_det * sizeof(float)));
    CHECK(cudaMemset(runner->d_input_, 0, max_batch_ * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(float)));
    std::memset(runner->h_output_, 0, max_batch_ * size  * sizeof(float));
    
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
    std::cout << "batch size " << batch_size << std::endl; // debug
    // 数据拷贝 htod
    timer.start();
    cudaMemcpyAsync(d_input_beforePre_, image, batch_size * 3 * image_width * image_height * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_htod += timer.last_elapsed();
    //Debug: check input image CHECK => Done
    // uint8_t* im = reinterpret_cast<uint8_t*>(image);
    // for(int i = 0; i < 3 * image_height * image_width; i++){
    //     std::cout << "index: " << i << ", real data:" << im[0] << std::endl;
    // }

    
    // 预处理
    timer.start();
    prerpocess(batch_size, image_width, image_height);
    CHECK(cudaStreamSynchronize(stream_));
    timer.stop();
    time_pre += timer.last_elapsed();

    // 预处理debug
    //Debug: 保存预处理后的图像，看看显示正确不
    // float* dst = nullptr;
    // cudaMallocHost(&dst, 3 * m_input_height * m_input_width * sizeof(float));
    // cudaMemcpy(dst, d_input_, 3 * m_input_height * m_input_width * sizeof(float), cudaMemcpyDeviceToHost);
    // std::ofstream outFile("./preprocess_img.bin", std::ios::binary);
    // if (!outFile.is_open()) {
    //     std::cerr << "无法打开文件" << std::endl;
    //     return 1;
    // }
    // for(int i = 0; i < 3 * m_input_height * m_input_width; i++){
    //     // std::cout << "preprocess data:" << dst[i]*255 << std::endl;
    //     float scaledData = dst[i] * 255.0f;
    //     uint8_t data = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, scaledData)));
    //     // std::cout << "real data:" << static_cast<int>(data) << std::endl;
    //     outFile.write(reinterpret_cast<const char*>(&data), sizeof(data));
    // }
    // if (!outFile.good()) {
    //     std::cerr << "写入文件时发生错误" << std::endl;
    //     return 1;
    // }
    // outFile.close();
    // cudaFree(dst);

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
    cudaMemcpyAsync(h_output_, d_output_post_, (m_max_det * 7 + 1) * batch_size, cudaMemcpyDeviceToHost, stream_);
    // cudaMemcpyAsync(h_output_, d_output_, out_size_ * batch_size, cudaMemcpyDeviceToHost, stream_);
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
    for( int index = 0; index < batch_size; index++){
        // 获取当前batch的起点地址
        uint8_t* src = reinterpret_cast<uint8_t*>(d_input_beforePre_) + index * src_offsize;
        float* dst = reinterpret_cast<float*>(d_input_) + index * dst_offsize;
        float scale =  std::min(float(m_input_width) / image_width, float(m_input_height) / image_height);
        int img_w = scale * image_width;
        int img_h = scale * image_height;
        int pad_w = (m_input_width - img_w) / 2;
        int pad_h = (m_input_height - img_h) / 2;
        // printf("img_w:%d, img_h:%d, pad_w:%d, pad_h:%d\n", img_w, img_h, pad_w, pad_h); // debug
        // printf("scale:%f, mean:%f, std:%f\n",scale_, mean_[0], std_[0]); // debug 
        // 详情用法可参考dlpreprocess/include/image_proc.h, false代表保持不改变RGB顺序。
        RGBROIBilinearResizeNormPadPlane(src, dst, image_width, image_height, m_input_width, m_input_height, img_w, img_h, pad_w, pad_h,
                                            0,  0, image_width, image_height, scale_, mean_[0], mean_[1], mean_[2], std_[0], std_[1], std_[2], pad_[0], pad_[1], pad_[2],
                                            true, stream_);
    }
}

void FrontDetectorRunner::postprocess(int batch_size){
    const int num_bboxes = outputDims_.d[0] * outputDims_.d[1];
    const int num_classes = outputDims_.d[2] - 5; // -5 是减去(x, y, w, h, conf)
    std::cout << "num bboxes output " << num_bboxes << ", num classes is " << num_classes << std::endl; // debug 
    std::cout << "confidence_threshold:" << m_conf_thres << ", nms_threshold:" << m_iou_thres << std::endl;

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