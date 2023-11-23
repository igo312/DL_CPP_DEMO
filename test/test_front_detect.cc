#include "dlnne_algo_front_detect.h"
#include "dl_timer.h"
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <cuda_runtime_api.h>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./test_front_detect [m] [b] [c]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为8" << std::endl;
    std::cout << "     c: 所有线程总的图片推理张数， 默认为6000" << std::endl;
    std::cout << "     t: 推理线程的数量，默认为1" << std::endl;
    std::cout << "     h: 显示此帮助信息" << std::endl;
    std::cout << "     eg: ./test_front_detect -m ./front_detect.onnx -b 8 -t 4 -c 6000" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, int &infer_count, int &thread_num){
    int opt = 0;
    const char* opt_string = "m:b:c:t:h";
    while( -1 != (opt = getopt(argc, argv, opt_string))){
        switch(opt){
            case 'm':
                model_path = optarg;
                break;
            case 'b':
                maxBatchSize = atoi(optarg);
                break;
            case 'c':
                infer_count = atoi(optarg);
                break;
            case 't':
                thread_num = atoi(optarg);
                break;
            default:
                showUsage();
                break;
        }
    }
}

int main(int argc, char* argv[]){
    std::string model_path = "";
    int maxBatchSize = 8;
    int infer_count = 6000;
    int thread_num = 1;
    getCustomerOpt(argc, argv, model_path, maxBatchSize, infer_count, thread_num);

    std::vector<std::string> node_names = {"tu1", "tu2_tiled_concate", "quantized_concat", "quantized_concat6", "quantized_concat13", "empty----"};
    auto frontdetectEngine = FrontDetectorBuilder(model_path, maxBatchSize, node_names);
    
    float* input;
    cudaMallocHost((void**)&input, maxBatchSize * frontdetectEngine.get_input_size());

    auto infer_func = [&frontdetectEngine, &input](int count, int batch_size){
        auto runner = frontdetectEngine.getRunner();
        for(int i = 0; i < count; i++){
            runner->infer(input, batch_size);
        }
        runner->print_timer();
    };
    
    std::vector<std::thread> threads;
    // int per_infer_count = infer_count / maxBatchSize / thread_num;
    int per_infer_count = infer_count / maxBatchSize;
    DlCpuTimer timer;

    timer.start();
    for(int i = 0; i < thread_num; i++){
        threads.push_back(std::thread(infer_func, per_infer_count, maxBatchSize));
    }

    for(int i = 0; i < thread_num; i++){
        threads[i].join();
    }
    timer.stop();

    printf("Total infer count: %d, thread num: %d, total time is %fs\n", infer_count, thread_num, timer.last_elapsed() / 1000);

    cudaFree(input);
    return 0;
}