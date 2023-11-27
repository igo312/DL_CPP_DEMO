#ifndef YUV_CROP_H_
#define YUV_CROP_H_

#include <stdio.h>
#include "common.h"

__global__ void yu12_crop_kernel(uint8_t *__restrict__ input, uint8_t *__restrict__ output,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w) {
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

    uint8_t *in_y = input;
    uint8_t *out_y = output;
    if (h_idx < out_h && x_idx < out_w) {
        if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w) {
          out_y[h_idx * out_w + x_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
        }
    }

    int in_uv_h = in_h >> 1;
    int in_uv_w = in_w >> 1;
    int out_uv_h = out_h >> 1;
    int out_uv_w = out_w >> 1;
    uint8_t *in_u = in_y + in_h * in_w;
    uint8_t *in_v = in_u + (in_h * in_w >> 2);
    uint8_t *out_u = out_y + out_h * out_w;
    uint8_t *out_v = out_u + (out_h * out_w >> 2);
    if (h_idx < out_uv_h && x_idx < out_uv_w) {
        if (((start_h >> 1) + h_idx < in_uv_h) && (start_w >> 1) + x_idx < in_uv_w) {
          int in_idx = ((start_h >> 1) + h_idx) * in_uv_w + (start_w >> 1) + x_idx;
          int out_idx = h_idx * out_uv_w + x_idx;
          out_u[out_idx] = in_u[in_idx];
          out_v[out_idx] = in_v[in_idx];
        }
    }
}

__global__ void nv12_crop_kernel(uint8_t *__restrict__ input, uint8_t *__restrict__ output,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w) {
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

    uint8_t *in_y = input;
    uint8_t *out_y = output;
    if (h_idx < out_h && x_idx < out_w) {
        if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w) {
          out_y[h_idx * out_w + x_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
        }
    }

    int in_uv_h = in_h >> 1;
    int in_uv_w = in_w >> 1;
    int out_uv_h = out_h >> 1;
    int out_uv_w = out_w >> 1;
    uint16_t *in_uv = (uint16_t *)(input + in_h * in_w);
    uint16_t *out_uv = (uint16_t *)(output + out_h * out_w);
    if (h_idx < out_uv_h && x_idx < out_uv_w) {
        if (((start_h >> 1) + h_idx < in_uv_h) && (start_w >> 1) + x_idx < in_uv_w) {
          out_uv[h_idx * out_uv_w + x_idx] = in_uv[((start_h >> 1) + h_idx) * in_uv_w + (start_w >> 1) + x_idx];
        }
    }
}

#endif

