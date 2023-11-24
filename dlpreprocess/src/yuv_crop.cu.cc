#include <stdio.h>
#include "yuv_crop.h"

void YU12Crop(uint8_t *in_buf, uint8_t *out_buf,
            int start_w, int start_h, int w_in,
            int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1), 1);
    yu12_crop_kernel<<<grid, block, 0, stream>>>(in_buf, out_buf, start_h, start_w, h_in, w_in, h_out, w_out);

}

void NV12Crop(uint8_t *in_buf, uint8_t *out_buf,
            int start_w, int start_h, int w_in, 
            int h_in, int w_out, int h_out, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid((w_out + block.x - 1) / block.x, (h_out + block.y - 1), 1);
    nv12_crop_kernel<<<grid, block, 0, stream>>>(in_buf, out_buf, start_h, start_w, h_in, w_in, h_out, w_out);

}
