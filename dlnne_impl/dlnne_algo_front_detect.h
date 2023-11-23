#include "dlnne_algo_unit.h"
#include "dl_timer.h"

using namespace dl::nne;
using namespace dl; 

class FrontDetectorBuilder : public NetworkBuilder{
    public:
        FrontDetectorBuilder(const std::string& model_path, int max_batch = 1, std::vector<std::string> subgraphs = {}, int device_id = 0): NetworkBuilder(model_path, max_batch, subgraphs, device_id){
            std::cout << "Front Detector Builder init" << std::endl;
        }

        std::shared_ptr<NetworkRunner> getRunner() override;

        int get_input_size() override;

   
};

class FrontDetectorRunner : public NetworkRunner{
    public:
        friend class FrontDetectorBuilder;

        FrontDetectorRunner(Engine* engine_, int device_id = 0):NetworkRunner(engine_, device_id){
            std::cout << "return a Front Detector Runner" << std::endl;
            timer = DlCpuTimer();
            reset_timer();
        }
        ~FrontDetectorRunner();
        void infer_async(float* image, int batch_size) override;
        void infer(float* image, int batch_size) override;

    private:
        void prerpocess(float* image, int batch_index) override;
        void postprocess(float* image, int batch_inddex) override;

        void* d_input_ = nullptr;
        void* d_output_ = nullptr;
        void* h_output_ = nullptr;
        
        // one batch input and output size
        int inp_size_;
        int out_size_;

        float mean_[3] = {0., 0., 0.};
        float std_[3] = {1., 1., 1.};

        Dims inputDims_;
        Dims outputDims_;

      
       
};