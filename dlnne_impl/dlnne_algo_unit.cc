#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include "cuda.h"
#include <dlnne/dlnne.h>
#include <cuda_runtime_api.h>
#include <string>
#include "dlnne_algo_unit.h"
#include "callback.h"
#include <fstream>

using namespace dl::nne;
 
NetworkBuilder::NetworkBuilder(const std::string& model_path, int max_batch, std::vector<std::string> subgraphs,  int device_id){
    std::cout << "model_path = " << model_path << std::endl; 
    max_batch_ = max_batch; 
    device_id_ = device_id;

    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    ClusterConfig config = kCluster0123;
    if ( 1 == prop.clusterCount){
        config = kCluster0;
    }else if( 2 == prop.clusterCount ){
        config = kCluster01;
    }

    bool serialized = (model_path.find(".engine") != std::string::npos);
    if (serialized){
        std::cout << "deserialize the engine from file: " << model_path << std::endl;
        std::ifstream slz(model_path);
        slz.seekg(0, std::ios::end);
        uint64_t length = static_cast<uint64_t>(slz.tellg());
        slz.seekg(0, std::ios::beg);

        char *slz_data = new char[length];
        slz.read(slz_data, static_cast<int64_t>(length));
        engine_ = Deserialize(slz_data, length);
        delete[] slz_data;
    }else{
        auto builder = CreateInferBuilder();
        auto network = builder->CreateNetwork();

        BuilderConfig builder_cfg;
        builder_cfg.callback = nullptr;
        builder_cfg.max_batch_size = max_batch_;
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
        engine_ = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();

    }
    auto nb_bindings = engine_->GetNbBindings();
    for(auto i = 0; i < nb_bindings; i++) {
            auto name = engine_->GetBindingName(i);
            auto shape = engine_->GetBindingDimensions(i);
            auto data_type = engine_->GetBindingDataType(i);
            bool is_input = engine_->BindingIsInput(i);

            std::cout  << (is_input ? "Input " : "Output ") 
                 << "binding name = " << name
                 << ", data type = " << data_type 
                 << " shape = [" ;

            for(int i = 0; i < shape.nbDims; i++) {
                if(i)
                    std::cout << ",";
                std::cout << shape.d[i];
            }
            std::cout << "]" << std::endl;
        }
}

NetworkBuilder::~NetworkBuilder(){
    if (engine_ != nullptr){
        engine_->Destroy();
        engine_ = nullptr;
    }
}

void NetworkBuilder::serializedEngine(const std::string& serialized_path){
    auto ser_res = engine_->Serialize();
    std::ofstream slz(serialized_path);
    slz.write(static_cast<char *>(ser_res->Data()),
              static_cast<int64_t>(ser_res->Size()));
    slz.close();
    ser_res->Destroy();
    std::cout << "serialized done, saved to " << serialized_path << std::endl;
}

NetworkRunner::NetworkRunner(Engine* engine_, int device_id=0){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    ClusterConfig config = kCluster0123;
    if ( 1 == prop.clusterCount){
        config = kCluster0;
    }else if( 2 == prop.clusterCount ){
        config = kCluster01;
    }
    context_ = engine_->CreateExecutionContext(config);
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

}

NetworkRunner::~NetworkRunner(){
    if(context_ != nullptr){
        context_->Destroy();
        context_ = nullptr;
    }

    if(stream_ != nullptr){
        cudaStreamDestroy(stream_);
    }
}