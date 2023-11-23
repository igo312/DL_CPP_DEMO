#include <string>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include "cuda.h"
#include <dlnne/dlnne.h>
#include <cuda_runtime_api.h>
#include "dl_timer.h"

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
        virtual void infer_async(float* image, int batch_size) = 0;
        virtual void infer(float* image, int batch_size) = 0;

        void reset_timer(){
            time_htod = 0;
            time_dtoh = 0;
            time_infer = 0;
            batch_count = 0;
        }

        void print_timer(){
            std::cout << "Front Detector Time consuming print:" << std::endl;
            std::cout << "Total infer image counts: " << batch_count << ", htod: " << time_htod / 1000 << "s, infer: " << time_infer / 1000 << "s, dtoh: " << time_dtoh / 1000 << "s." << std::endl;
        }

    protected:
        virtual void prerpocess(float* image, int batch_index) = 0;
        virtual void postprocess(float* image, int batch_inddex) = 0;
        ExecutionContext* context_ = nullptr;
        cudaStream_t stream_ = nullptr;
        int m_input_width = 0;
        int m_input_height = 0;

        float time_htod;
        float time_dtoh;
        float time_infer;
        int batch_count;
        DlCpuTimer timer; 
};