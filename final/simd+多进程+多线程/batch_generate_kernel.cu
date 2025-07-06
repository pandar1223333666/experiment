#include <cuda_runtime.h>
#include <string.h>
#include "gpu_structs.h"

__device__ const char* get_value_ptr(const char* all_values, int offset, int idx) {
    int pos = offset;
    for (int i = 0; i < idx; ++i) {
        while (all_values[pos] != '\0') ++pos;
        ++pos;
    }
    return &all_values[pos];
}

__global__ void batch_generate_kernel(
    PT_GPU* pts, Segment_GPU* letters, Segment_GPU* digits, Segment_GPU* symbols,
    char* all_values,
    char* output, int* output_offsets, int* pt_offsets, int batch_size, int total_combos)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_combos) return;

    // 1. 找到属于哪个PT
    int pt_idx = 0;
    while (pt_idx < batch_size && global_idx >= pt_offsets[pt_idx + 1]) ++pt_idx;
    if (pt_idx >= batch_size) return;

    int combo_idx = global_idx - pt_offsets[pt_idx];
    PT_GPU pt = pts[pt_idx];

    // 2. combo_idx解码为curr_indices
    int curr_indices[MAX_SEG];
    int div = combo_idx;
    for (int i = pt.seg_num - 1; i >= 0; --i) {
        curr_indices[i] = div % pt.max_indices[i];
        div /= pt.max_indices[i];
    }

    // 3. 拼接guess
    char guess[128] = {0}; // 假设最大口令长度64
    int guess_len = 0;
    for (int i = 0; i < pt.seg_num; ++i) {
        int type = pt.types[i];
        int value_idx = curr_indices[i];
        int type_idx = pt.type_indices[i];
        const char* seg_val = nullptr;

        if (type == 1 && type_idx >= 0)
            seg_val = get_value_ptr(all_values, letters[type_idx].value_offset, value_idx);
        else if (type == 2 && type_idx >= 0)
            seg_val = get_value_ptr(all_values, digits[type_idx].value_offset, value_idx);
        else if (type == 3 && type_idx >= 0)
            seg_val = get_value_ptr(all_values, symbols[type_idx].value_offset, value_idx);

        if (seg_val) {
            int l = 0;
            while (seg_val[l] && guess_len < 63) {
                guess[guess_len++] = seg_val[l++];
            }
        }
    }

    // 写入输出
    int offset = output_offsets[global_idx];
    for (int i = 0; i < guess_len; ++i) {
        output[offset + i] = guess[i];
    }
    output[offset + guess_len] = '\0';
}

extern "C" void launch_batch_generate_kernel(
    PT_GPU* pts, Segment_GPU* letters, Segment_GPU* digits, Segment_GPU* symbols,
    char* all_values,
    char* output, int* output_offsets, int* pt_offsets, int batch_size, int total_combos,
    int threads, int blocks)
{
    batch_generate_kernel<<<blocks, threads>>>(
        pts, letters, digits, symbols, all_values, output, output_offsets, pt_offsets, batch_size, total_combos);
    cudaDeviceSynchronize();
}