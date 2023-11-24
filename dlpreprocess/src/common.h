#ifndef COMMON_H_
#define COMMON_H_

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a):(b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a):(b))
#define MIN_3(a,b,c) MIN(MIN(a,b),(c))
#endif

#ifndef CLIP
#define CLIP(a,start,end) MAX(MIN((a),(end)),(start))
#endif

#define ALIGN1B 0
#define ALIGN2B 1
#define ALIGN4B 2
#define ALIGN8B 3


// ((uint64_t)global & 7) == 0 and (size & 7) == 0
__device__ __forceinline__ void global2share_copy_align(const uint8_t *global, uint8_t *sm, int size) {
    int idx = threadIdx.x;
    int copy_num = size >> 3;
    while (idx < copy_num) {
        ((uint64_t*)sm)[idx] = ((uint64_t*)global)[idx];
        idx += blockDim.x;
    }
}

/*
Usage:
int offset = global2share_copy(global, sm, size);   
sm += sm;
*/
__device__ __forceinline__ uint8_t global2share_copy(const uint8_t *global, uint8_t *sm, int size) {
    int idx = threadIdx.x;
    // &7 表示对8取余数，所有的threadIdx.x 共享front和back
    uint8_t front = (8 - ((uint64_t)global & 7)) & 7; // 使得拷贝起始地址是8的倍数
    uint8_t back = (size - front) & 7;               // 使拷贝的个数是8的倍数
    // 向右移三位，表示除8，八字节拷贝比较快
    int copy_num = (size - front - back) >> 3;
    // 加1是在uint64_t 是偏移量，留八个字节是为了保留偏移量的数据 => 拷贝的数据是前后不对齐，中间对齐都是8的倍数拷贝
    while (idx < copy_num) {
        ((uint64_t*)sm)[idx + 1] = ((uint64_t*)(global + front))[idx];
        idx += blockDim.x;
    }
    // 只有前几个threadIdx和后面几个需要加
    if (threadIdx.x < front) {
        sm[8 - front + threadIdx.x] = global[threadIdx.x];
    }
    if (threadIdx.x < back) {
        sm[8 + (copy_num << 3) + threadIdx.x] = global[size - back + threadIdx.x];
    }
    return 8 - front; //尾部的data
}

#define CUDA_CHECK(state)                                                       \
    do {                                                                        \
      if (state != cudaSuccess) {                                               \
        std::cout << "CUDA Error code num is:" << state << std::endl;           \
        std::cout << "CUDA Error:" << cudaGetErrorString(state) << std::endl;   \
        std::cout << __FILE__ << " " << __LINE__ << "line!" << std::endl;       \
        abort();                                                                \
      }                                                                         \
    } while (0)
#endif /* COMMON_H_ */
