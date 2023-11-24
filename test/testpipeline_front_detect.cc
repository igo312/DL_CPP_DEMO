#include "dl_timer.h"
#include "dlnne_algo_unit.h"
#include <opencv2/opencv.hpp>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./testpipeline_front_detect [m] [b] [p]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为1" << std::endl;
    std::cout << "     p: 图片地址" << std::endl;
    std::cout << "     h: 显示此帮助信息" << std::endl;
    std::cout << "     eg: ./testpipeline_front_detect -m ./front_detect.onnx -b 1 -p ${image_path}" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, std::string &image_path){
    int opt = 0;
    const char* opt_string = "m:b:p:h";
    while( -1 != (opt = getopt(argc, argv, opt_string))){
        switch(opt){
            case 'm':
                model_path = optarg;
                break;
            case 'b':
                maxBatchSize = atoi(optarg);
                break;
            case 'p':
                image_path = optarg;
                break;
            default:
                showUsage();
                break;
        }
    }
}

int main(int argc char* argv[]){
    std::string model_path = "";
    std::string image_path = "";
    int max_batchsize = 1; 
    getCustomerOpt(argc, argv, model_path, max_batchsize, image_path);
    
}