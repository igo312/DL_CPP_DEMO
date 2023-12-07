# 登临 网络推理 C++ Demo
# 快速执行
```
# 包含了编译、跑网络pipeline的过程
bash run.sh
```
# 1. 代码结构
有四个部分组成，依次进行介绍其作用

1. `dlpreprocess` : 各类前处理后处理的cuda kernel，可以通过头文件[`image.proc.h`](./dlpreprocess/include/image_proc.h)查阅。需要提前编译。
2. `dlutil` : 提供计时和子图划分的callback。
3. `dlnne_impl`: 网络的实现，demo提供了front-detect的实现
   1. `dlnne_algo_unit`中实现了Builder和Runner，各网络实例将继承于此进行实现
   2. `dlnne_algo_front_detect`： 主要实现了runner类的各个方法
      1. `execute` 带htod，dtoh，网络推理，与当前的测试一致
      2. `infer` 则是完整的pipeline，包括了数据拷贝、前处理、网络推理、后推理等
      3. 带async的是将所有操作放在同一个流上，在函数最后进行同步。

4. `test`: 提供了网络性能测试、网络的序列化、网络性能的测试、网络pipeline四个功能。
   1. `testnetworkperf`: 纯网络性能测试
   2. `testserialize`: 模型序列化，直接调用sdk原生的接口，可用于学习sdk api。
   3. `testperformance_frontDetect`: 包含htod dtoh的数据传输、以及网络推理的性能测试。
   4. `testpipeline_frontDetect`: 对单张图片完整流程的推理，包含前处理、后处理。

# 2. 编译
1. 编译 dlpreprocess
2. 编译其余代码
   
可参考[`run.sh`](./run.sh)的编译过程

# Q&A

1. 
```
Q：为什么执行run.sh生成bmp文件结果是倒置的？
A: 因为原图是bgr，导致了这个结果。
```

2. 
```
Q：是否可以添加OpenCV的支持
A: 目前testpipeline_frontDetect 支持了OpenCV，但是注释了，如果需要可以放开，参考使用。
```

   


   