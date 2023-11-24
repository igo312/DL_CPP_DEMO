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

// 已假设真实输入是uint8型，onnx输入类型是float32型
// 默认模型输出只有一个,输入[1,3,640,640],输出[1,25200,11]
class FrontDetectorRunner : public NetworkRunner{
    public:
        friend class FrontDetectorBuilder;

        FrontDetectorRunner(Engine* engine_, int device_id = 0):NetworkRunner(engine_, device_id){
            std::cout << "return a Front Detector Runner" << std::endl;
            timer = DlCpuTimer();
            reset_timer();
        }
        ~FrontDetectorRunner();

        void infer_async(void* image, int batch_size, int image_width, int image_height) override;
        void infer(void* image, int batch_size, int image_width, int image_height) override;

        void execute_async(void* image, int batch_size) override;
        void execute(void* image, int batch_size) override;

        void* return_output() override{
            return h_output_;
        }

    private:
        void prerpocess(int batch_size, int image_width, int image_height) override;
        void postprocess(int batch_size) override;

        
        float mean_[3] = {0., 0., 0.};
        float std_[3] = {255., 255., 255.};
        float pad_[3] = {114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f};
        float scale_ = 1.0f / 255.0f;
        
        // onnx的模型输入大小
        int m_input_width = 640;
        int m_input_height = 640;

        // onnx的模型输出大小 one batch
        int out_size_; // onnx输出的大小总和 byte
        int out_size_post_; // 经过后处理后的大小

        // 输入输出的device指针，由于不同模型的输出个数不同，因此是实例类的属性
        void* d_input_ = nullptr; // 模型的真实输入指针，与onnx的输入对齐
        void* d_input_beforePre_ = nullptr; // 预处理之前的输入,目前应该是int8类型的指针
        void* d_output_ = nullptr; 
        void* d_output_post_ = nullptr; // 经过后处理之后，在本例中是经过nms之后的。
        void* h_output_ = nullptr; // 经过后处理之后的内存拷出

        
        // 后处理超参
        float m_conf_thres = 0.25;
        float m_iou_thres = 0.45;
        int m_max_det = 1000;

        // engine的输入输出信息
        Dims inputDims_;
        Dims outputDims_;
       
};