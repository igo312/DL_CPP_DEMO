#include "dlnne_algo_unit.h"
#include <unistd.h>
#include <string>
#include <vector>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./test_front_detect [m] [b] [f]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为8" << std::endl;
    std::cout << "     f: 输出的序列化engine文件路径" << std::endl;
    std::cout << "     eg: ./testserialize -m ./front_detect.onnx -b 8 -f ./test.engine" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, std::string &engine_path){
    int opt = 0;
    const char* opt_string = "m:b:f:h";
    while( -1 != (opt = getopt(argc, argv, opt_string))){
        switch(opt){
            case 'm':
                model_path = optarg;
                break;
            case 'b':
                maxBatchSize = atoi(optarg);
                break;
            case 'f':
                engine_path = optarg;
                break;
            default:
                showUsage();
                break;
        }
    }
}

int main(int argc, char* argv[]){
    std::string model_path = "";
    std::string engine_path = "";
    int max_batch = 8;
    getCustomerOpt(argc, argv, model_path, max_batch, engine_path);

    int device_id = 0;

    std::vector<std::string> node_names = {"tu1", "tu2_tiled_concate", "quantized_concat", "quantized_concat6","quantized_concat10", "quantized_concat13", "empty----"};
    NetworkBuilder engine(model_path, 8, node_names, device_id);
    engine.serializedEngine(engine_path);
    printf("model:%s serialized to engine:%s done", model_path.c_str(), engine_path.c_str());
}