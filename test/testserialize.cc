#include "dlnne_algo_unit.h"
#include <unistd.h>
#include <string>
#include <vector>
#include <cstring>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./test_front_detect [m] [b] [f] [g]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为8" << std::endl;
    std::cout << "     f: 输出的序列化engine文件路径" << std::endl;
    std::cout << "     g: 传入子图" << std::endl;
    std::cout << "     h: 显示此帮助信息" << std::endl;
    std::cout << "     eg: ./testserialize -m ./front_detect.onnx -b 8 -f ./test.engine" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, std::string &engine_path, std::vector<std::string>& subgraphs){
    int opt = 0;
    const char* opt_string = "m:b:f:g:h";
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
            case 'g':
            {
                char* token = std::strtok(optarg, ",");
                while( token != nullptr){
                    std::string s(token);
                    s.erase(0,s.find_first_not_of(" "));
                    s.erase(s.find_last_not_of(" ") + 1);
                    subgraphs.push_back(s);
                    token = std::strtok(nullptr, ",");
                }
            }
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
    std::vector<std::string> node_names = {};
    getCustomerOpt(argc, argv, model_path, max_batch, engine_path, node_names);

    int device_id = 0;

    
    NetworkBuilder engine(model_path, max_batch, node_names, device_id);
    engine.serializedEngine(engine_path);
    printf("maxbatch is %d, model:%s serialized to engine:%s done\n", max_batch, model_path.c_str(), engine_path.c_str());
}