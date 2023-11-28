#include "dl_timer.h"
#include "dlnne_algo_front_detect.h"
// #include <opencv2/opencv.hpp>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <math.h>

using namespace dl;

void showUsage(){
    std::cout << "Usage: ./testpipeline_front_detect [m] [b] [p] [c] [s]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型max batch size， 默认为1" << std::endl;
    std::cout << "     p: 图片地址，目前仅支持二进制保存的文件" << std::endl;
    std::cout << "     c: 模型输出的类别数目" << std::endl;
    std::cout << "     s: 输入二进制文件宽度x高度， 默认值为1920x1080";
    std::cout << "     h: 显示此帮助信息" << std::endl;
    std::cout << "     eg: ./testpipeline_front_detect -m ./front_detect.onnx -b 1 -p ${image_path} -c 80 -s 1920x1080" << std::endl; 
    exit(0);
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, std::string &image_path, int &num_classes, int &width, int &height){
    int opt = 0;
    const char* opt_string = "m:b:p:c:s:h";
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
            case 's':
            {
                char* p = std::strtok(optarg, "x");
                width = atoi(p);
                p = std::strtok(NULL, " ");
                height = atoi(p);
            }
                break;
            default:
                showUsage();
                break;
        }
    }
}
int map_to_origin_img(float value, int pad, float scale, float min_=0, float max_=2048);
void load_image(char* &buffer, const std::string image_path, const int width, const int height);
void save_bmp(char* buffer, const std::string save_path, const int width, const int height);

int main(int argc, char* argv[]){
    std::string model_path = "";
    std::string image_path = "";
    int max_batchsize = 1; 
    int num_classes = 80;
    int width = 1920;
    int height = 1080;
    getCustomerOpt(argc, argv, model_path, max_batchsize, image_path, num_classes, width, height);
    
    // 读取图片
    char* buffer = nullptr;
    load_image(buffer, image_path, width, height);

    // 推理并返回结果
    auto frontdetectEngine = FrontDetectorBuilder(model_path, max_batchsize);
    auto runner = frontdetectEngine.getRunner();
    runner->infer(buffer, 1, width, height);
    auto output = (float*)runner->return_output();
    
    // 画图，假设已知模型的输入是640,640，这样方便计算scale
    // 用置信度判断该框是否保留率
    float scale = std::min(640.0 / (float)width, 640.0 / (float)height);
    int pad_w = (640 - width*scale)/2;
    int pad_h = (640 - height*scale)/2; 
    int output_num = (int)output[0];
    output = output + 1;
    printf("output %d boxes\n", output_num);
    int stride = width * height; // 一个通道的数目
    for(int i = 0; i < output_num; i++){
        float x1 = output[0];
        float y1 = output[1];
        float x2 = output[2];
        float y2 = output[3];
        float conf = output[4];
        float classid = output[5];
        bool keep_flag = output[6] > 0 ? true : false;
        // std::cout << keep_flag << std::endl;
        if(keep_flag){
             // 预测还原到原图上
            int x1_ = map_to_origin_img(x1, pad_w, scale, 0, width);
            int x2_ = map_to_origin_img(x2, pad_w, scale, 0, width);
            int y1_ = map_to_origin_img(y1, pad_h, scale, 0, height);
            int y2_ = map_to_origin_img(y2, pad_h, scale, 0, height);
            printf("x1:%d, y1:%d, x2:%d, y2:%d, conf:%.4f, classid:%.1f\n", x1_, y1_, x2_, y2_, conf, classid);
            // 画图
            // cv::Rect roi(x1_, y1_, x2_ - x1_, y2_ - y1_);
            // cv::rectangle(image, roi, cv::Scalar(0,0,255), 2);
            /* 上下两条横边*/
            for(int i = 0; i < x2_ - x1_; i++){
                // 使用y1_ 确定上边
                buffer[(y1_ * width + x1_ + i) * 3 + 0] = 0x00; // Blue channel
                buffer[(y1_ * width + x1_ + i) * 3 + 1] = 0x00; // Green channel
                buffer[(y1_ * width + x1_ + i) * 3 + 2] = 0xff; // Red channel
                // 使用y2_ 确定下边
                buffer[(y2_ * width + x1_ + i) * 3 + 0] = 0x00; // Blue channel
                buffer[(y2_ * width + x1_ + i) * 3 + 1] = 0x00; // Green channel
                buffer[(y2_ * width + x1_ + i) * 3 + 2] = 0xff; // Red channel
            }
            /* 左右两条横边*/
            for(int i = 0; i < y2_ - y1_; i++){
                // 使用x1_ 确定左边
                buffer[((y1_+i) * width + x1_) * 3 + 0] = 0x00; // Blue channel
                buffer[((y1_+i) * width + x1_) * 3 + 1] = 0x00; // Green channel
                buffer[((y1_+i) * width + x1_) * 3 + 2] = 0xff;  // Red channel
                // 使用x2_ 确定右边
                buffer[((y1_+i) * width + x2_) * 3 + 0] = 0x00; // Blue channel
                buffer[((y1_+i) * width + x2_) * 3 + 1] = 0x00; // Green channel
                buffer[((y1_+i) * width + x2_) * 3 + 2] = 0xff; // Red channel
            }

        }
       
        output = output +  7; // 7 is a box size, it can find non_maximum_suppresion function in the dlpreprocess include file .
    }
    
    // 保存图片
    // cv::imwrite("./output.jpg", image);
    // save to bmp, because output is bgr order, so the image will flip
    save_bmp(buffer, "./result.bmp", width, height);
    // save to bgr
    std::ofstream outFile("./result.bgr", std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    for(int i = 0; i < 3 * height * width; i++){
        outFile.write(buffer+i, sizeof(uint8_t));
    }
    if (!outFile.good()) {
        std::cerr << "写入文件时发生错误" << std::endl;
        return 1;
    }
    outFile.close();

    delete[] buffer;
    
}

int map_to_origin_img(float value, int pad, float scale, float min_=0, float max_){
    value = (value - pad) / scale;
    value = std::max(value, min_);
    value = std::min(value, max_);
    return std::round(value);
}

void load_image(char*& buffer, const std::string image_path, const int width, const int height){
    // cv::Mat image = cv::imread(image_path);
    std::ifstream file(image_path, std::ios::binary);
    if (file.is_open()) {
        // 获取文件大小
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // 分配内存来存储文件内容
        buffer = new char[fileSize];

        // 读取文件内容到缓冲区
        file.read(buffer, fileSize);
        std::cout << "FileSize: "<< fileSize <<  ", Image read from: " << image_path << ", width:" << width << ", height:" << height << std::endl;

        // 关闭文件
        file.close();
    } else {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
}

typedef struct                       /**** BMP file header structure ****/  
{  
    unsigned int   bfSize;           /* Size of file */  
    unsigned short bfReserved1;      /* Reserved */  
    unsigned short bfReserved2;      /* ... */  
    unsigned int   bfOffBits;        /* Offset to bitmap data */  
} MyBITMAPFILEHEADER;

typedef struct                       /**** BMP file info structure ****/  
{  
    unsigned int   biSize;           /* Size of info header */  
    int            biWidth;          /* Width of image */  
    int            biHeight;         /* Height of image */  
    unsigned short biPlanes;         /* Number of color planes */  
    unsigned short biBitCount;       /* Number of bits per pixel */  
    unsigned int   biCompression;    /* Type of compression to use */  
    unsigned int   biSizeImage;      /* Size of image data */  
    int            biXPelsPerMeter;  /* X pixels per meter */  
    int            biYPelsPerMeter;  /* Y pixels per meter */  
    unsigned int   biClrUsed;        /* Number of colors used */  
    unsigned int   biClrImportant;   /* Number of important colors */  
} MyBITMAPINFOHEADER;



void save_bmp(char* buffer, const std::string save_path, const int width, const int height){  
    MyBITMAPFILEHEADER bfh;  
    MyBITMAPINFOHEADER bih;  
    /* Magic number for file. It does not fit in the header structure due to alignment requirements, so put it outside */  
    unsigned short bfType=0x4d42;             
    bfh.bfReserved1 = 0;  
    bfh.bfReserved2 = 0;  
    bfh.bfSize = 2 + sizeof(MyBITMAPFILEHEADER) + sizeof(MyBITMAPINFOHEADER) + width* height * 3;  
    bfh.bfOffBits = 0x36;  
  
    bih.biSize = sizeof(MyBITMAPINFOHEADER);  
    bih.biWidth = width;  
    bih.biHeight = height;  
    bih.biPlanes = 1;  
    bih.biBitCount = 24;  
    bih.biCompression = 0;  
    bih.biSizeImage = 0;  
    bih.biXPelsPerMeter = 5000;  
    bih.biYPelsPerMeter = 5000;  
    bih.biClrUsed = 0;  
    bih.biClrImportant = 0;  
  
    FILE *file = fopen(save_path.c_str(), "wb");  
    if (!file)  
    {  
        printf("Could not write file\n");  
        return;  
    }  
  
    /*Write headers*/  
    fwrite(&bfType,sizeof(bfType),1,file);  
    fwrite(&bfh,sizeof(bfh),1, file);  
    fwrite(&bih,sizeof(bih),1, file);  
  
    fwrite(buffer,width*height*3,1,file);  
    fclose(file);  
}