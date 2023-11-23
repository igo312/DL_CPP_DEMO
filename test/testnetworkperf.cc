#include "dlnne_algo_unit.h"
#include <unistd.h>
#include <string>
#include <cstring>
#include <vector>
#include "dl_timer.h"
#include <fstream>
#include "callback.h"
using namespace dl;

void showUsage(){
    std::cout << "Usage: ./test_front_detect [m] [b] [c] [g]" << std::endl;
    std::cout << "     m: 模型文件路径" << std::endl;
    std::cout << "     b: 模型最大batch， 默认为8" << std::endl;
    std::cout << "     c: 测试图片数目" << std::endl;
    std::cout << "     g: 子图划分的例子" << std::endl;
    std::cout << "     eg: ./testnetworkperf -m ./front_detect.onnx -b 8 -c 100 -g \"tu1, tu2_tiled_concate, quantized_concat, quantized_concat6, quantized_concat13, empty----\" " << std::endl; 
}

void getCustomerOpt(int argc, char* argv[], std::string &model_path, int &maxBatchSize, int &count, std::vector<std::string>& subgraphs){
    int opt = 0;
    const char* opt_string = "m:b:c:g:h";
    while( -1 != (opt = getopt(argc, argv, opt_string))){
        switch(opt){
            case 'm':
                model_path = optarg;
                break;
            case 'b':
                maxBatchSize = atoi(optarg);
                break;
            case 'c':
                count = atoi(optarg);
                break;
            case 'g':
            {
                std::cout << "go th g " << std::endl;
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
            case 'h':
                showUsage();
                exit(0);
            default:
                showUsage();
                exit(1);
        }
    }
}

size_t getElementSize(DataType data_type){
    switch (data_type)
    {
    case kINT8:
    case kUINT8:
        return 1;
    case kINT16:
    case kUINT16:
    case kFLOAT16:
        return 2;
    case kFLOAT32:
    case kINT32:
    case kUINT32:
        return 4;
    case kFLOAT64:
    case kINT64:
    case kUINT64:
        return 8;
    default:
        std::cout << "Unsupport data type = " << data_type << std::endl;
    }
}

int main(int argc, char* argv[]){
    std::string model_path = "";
    int max_batch = 8;
    int count = 100;
    std::vector<std::string> subgraphs = {};
    getCustomerOpt(argc, argv, model_path, max_batch, count, subgraphs);
    DlCpuTimer timer;

    int device_id = 0;
  
    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    ClusterConfig config = kCluster0123;
    if ( 1 == prop.clusterCount){
        config = kCluster0;
    }else if( 2 == prop.clusterCount ){
        config = kCluster01;
    }
    Engine* engine = nullptr; 

    bool serialized = (model_path.find(".engine") != std::string::npos);
    if (serialized){
        std::cout << "deserialize the engine from file: " << model_path << std::endl;
        std::ifstream slz(model_path);
        slz.seekg(0, std::ios::end);
        uint64_t length = static_cast<uint64_t>(slz.tellg());
        slz.seekg(0, std::ios::beg);

        char *slz_data = new char[length];
        slz.read(slz_data, static_cast<int64_t>(length));
        engine = Deserialize(slz_data, length);
        delete[] slz_data;
    }else{
        auto builder = CreateInferBuilder();
        auto network = builder->CreateNetwork();

        BuilderConfig builder_cfg;
        builder_cfg.callback = nullptr;
        builder_cfg.max_batch_size = max_batch;
        builder_cfg.dump_dot = false;
        builder_cfg.print_profiling = false;
        
        CallbackImpl* callback = nullptr;
        if(!subgraphs.empty()){
            SubGraphGenFunc func = [subgraphs](const IDAG *graph,
                                    ISubgraphContainer *container){
                MergeNodes(graph, container, subgraphs);
                return true;
            };
            callback = new CallbackImpl();
            callback->setSubGraphGenerateCallback(func);
        }
        builder_cfg.callback = callback;

        WeightShareMode weight_share_mode =
            static_cast<WeightShareMode>(prop.clusterCount);
        builder_cfg.ws_mode = weight_share_mode;
        builder->SetBuilderConfig(builder_cfg);

        auto parser = dl::nne::CreateParser();

        std::cout <<" build engine from file: " << model_path << std::endl;
        parser->Parse(model_path.c_str(), *network);
        engine = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
    }
    auto context = engine->CreateExecutionContext(config);
    cudaStream_t stream; 
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    int nb_bindings = engine->GetNbBindings();
    std::vector<void *> bindings(nb_bindings);
    for(int i = 0; i < nb_bindings; i++){
        int size = max_batch;
        auto shape = engine->GetBindingDimensions(i);
        for(int j = 0; j < shape.nbDims; j++){
            size *= shape.d[j];
        }
        size *= getElementSize(engine->GetBindingDataType(i));
        cudaMalloc(&bindings[i], size);
        printf("binding name is %s, size is %d\n", engine->GetBindingName(i), size);
    }

    timer.start();
    for(int i = 0; i < count; i++){
        context->Enqueue(max_batch, bindings.data(), stream, nullptr);
        cudaStreamSynchronize(stream);
    }
    timer.stop();

    std::cout << "model_path: " << model_path <<
                 ", infer count: " << count <<
                 ", time elapsed: " << timer.last_elapsed()/1000 << "s, FPS:" << max_batch * count / ( timer.last_elapsed()/1000) << std::endl;

    if( engine != nullptr){
        engine->Destroy();
        engine = nullptr;
    }

    if( context != nullptr ){
        context->Destroy();
    }

    for(auto binding : bindings){
        cudaFree(binding);
    }
}