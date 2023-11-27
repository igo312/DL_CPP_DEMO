#include "dl_timer.h"
#include "dlnne_algo_front_detect.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./testpipeline_front_detect [m] [b] [p] [c]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为1" << std::endl;
    std::cout << "     p: 图片地址" << std::endl;
    std::cout << "     c: 模型输出的类别数目" << std::endl;
    std::cout << "     h: 显示此帮助信息" << std::endl;
    std::cout << "     eg: ./testpipeline_front_detect -m ./front_detect.onnx -b 1 -p ${image_path} -c 80" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, std::string &image_path, int &num_classes){
    int opt = 0;
    const char* opt_string = "m:b:p:c:h";
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
            case 'c':
                num_classes = atoi(optarg);
                break;
            default:
                showUsage();
                break;
        }
    }
}

int map_to_origin_img(float value, int pad, float scale, float min_=0, float max_=2048){
    value = (value - pad) / scale;
    value = std::max(value, min_);
    value = std::min(value, max_);
    return std::round(value);
}

int main(int argc, char* argv[]){
    std::string model_path = "";
    std::string image_path = "";
    int max_batchsize = 1; 
    int num_classes = 80;
    getCustomerOpt(argc, argv, model_path, max_batchsize, image_path, num_classes);
    
    // 读取图片
    cv::Mat image = cv::imread(image_path);
    int height = image.rows;
    int width = image.cols;
    if (image.empty()){
        std::cerr << "Error: Unable to read the image at " << image_path << std::endl;
        return 1; 
    }else{
        std::cout << "Image read from: " << image_path << ", width:" << width << ", height:" << height << std::endl;
    }

    // 推理并返回结果
    auto frontdetectEngine = FrontDetectorBuilder(model_path, max_batchsize);
    auto runner = frontdetectEngine.getRunner();
    runner->infer(image.data, 1, width, height);
    auto output = (float*)runner->return_output();
    
    // 画图，假设已知模型的输入是640,640，这样方便计算scale
    // 用置信度判断该框是否保留率
    float scale = std::min(640.0 / (float)width, 640.0 / (float)height);
    int pad_w = (640 - width*scale)/2;
    int pad_h = (640 - height*scale)/2; 
    int output_num = (int)output[0];
    output = output + 1;
    printf("output %d boxes\n", output_num);
    for(int i = 0; i < output_num; i++){
        float x1 = output[0];
        float y1 = output[1];
        float x2 = output[2];
        float y2 = output[3];
        float conf = output[4];
        float classid = output[5];
        bool keep_flag = output[6]>0?true:false;
        // std::cout << keep_flag << std::endl;
        if(keep_flag){
             // 预测还原到原图上
            int x1_ = map_to_origin_img(x1, pad_w, scale, 0, width);
            int x2_ = map_to_origin_img(x2, pad_w, scale, 0, width);
            int y1_ = map_to_origin_img(y1, pad_h, scale, 0, height);
            int y2_ = map_to_origin_img(y2, pad_h, scale, 0, height);
            printf("x1:%d, y1:%d, x2:%d, y2:%d, conf:%.4f, classid:%.1f\n", x1_, y1_, x2_, y2_, conf, classid);
            cv::Rect roi(x1_, y1_, x2_ - x1_, y2_ - y1_);
            cv::rectangle(image, roi, cv::Scalar(0,0,255), 2);
        }
       
        output = output +  7; // 7 is a box size, it can find non_maximum_suppresion function in the dlpreprocess include file .
    }
    cv::imwrite("./output.jpg", image);

    
}