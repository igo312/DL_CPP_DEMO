#include <string>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include "cuda.h"
#include <dlnne/dlnne.h>
#include <cuda_runtime_api.h>
#include "dl_timer.h"
#include <functional>

using namespace dl::nne;
using namespace dl;

class NetworkRunner;

class NetworkBuilder{
    public:
        NetworkBuilder(const std::string& model_path, int max_batch = 1, std::vector<std::string> subgraphs = {}, int device_id = 0);
        ~NetworkBuilder();

        virtual std::shared_ptr<NetworkRunner> getRunner(){};
        virtual int get_input_size(){};
        void serializedEngine(const std::string& serialized_path);
    
    protected:
        Engine* engine_ = nullptr;
        int max_batch_ = 1;
        int device_id_ = 0;

};

class NetworkRunner{
    public:
        NetworkRunner(Engine* engine_, int device_id=0);
        ~NetworkRunner();

        // infer include HostToDevice DeviceToHost data copy, model inference, preprocess, postprocess
        // infer 系列的函数包括数据拷贝、预处理、推理、后处理
        // 目前默认输入是图像3通道的，因此只用传入宽高即可
        // image: 真实的输入
        virtual void infer_async(void* image, int batch_size, int image_width, int image_height) = 0;
        virtual void infer(void* image, int batch_size, int image_width, int image_height) = 0;

        // execute include HostToDevice DeviceToHost data copy and model inference 
        // image: 默认输入的图像已经处理好
        virtual void execute_async(void* image, int batch_size) = 0;
        virtual void execute(void* image, int batch_size) = 0;
        virtual void* return_output() = 0;
        void reset_timer(){
            time_htod = 0;
            time_dtoh = 0;
            time_infer = 0;
            time_pre = 0;
            time_post = 0;
            batch_count = 0;
        }

        void print_timer(){
            std::cout << "Front Detector Time consuming print:" << std::endl;
            std::cout << "Total infer image counts: " << batch_count << ", htod: " << time_htod / 1000 << "s, infer: " << time_infer / 1000 << "s, dtoh: " << time_dtoh / 1000 << "s, ";
            std::cout << "preprocess: " << time_pre << "s, postprocess: " << time_post << "s." << std::endl;
        }

        // void measureTime(const std::function<void()>& operation, float& elapsedTime){
        //     timer.start();
        //     operation();
        //     cudaStreamSynchronize(stream_);
        //     timer.stop()
        //     elapsedTime += timer.last_elapsed();
        // }

    protected:
        // 默认预处理和后处理是异步的
        virtual void prerpocess(int batch_size, int image_width, int image_height) = 0;
        virtual void postprocess(int batch_size) = 0;
        ExecutionContext* context_ = nullptr;
        cudaStream_t stream_ = nullptr;
        
         // one batch input and output size
        int inp_size_beforePre_; // 系统输入的真实大小，即预处理前的大小
        int inp_size_; // onnx 输入的真实大小
    
        
        // 计时用的变量
        float time_htod;
        float time_dtoh;
        float time_infer;
        float time_pre;
        float time_post;
        int batch_count;
        DlCpuTimer timer; 
        
        

};