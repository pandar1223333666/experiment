#include <cuda_runtime.h>
#include <string.h>

#define MAX_LEN 64

__global__ void generate_guesses_kernel(const char* values, int value_len, int value_count, char* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < value_count) {
        // 拷贝value到输出
        for (int j = 0; j < value_len; ++j)
            out[idx * MAX_LEN + j] = values[idx * value_len + j];
        out[idx * MAX_LEN + value_len] = '\0';
    }
}


__global__ void generate_guesses_with_prefix_kernel(
    const char* prefix, int prefix_len,
    const char* values, int value_len,
    int value_count, char* out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < value_count) {
        // 拼接prefix
        for (int i = 0; i < prefix_len; ++i)
            out[idx * MAX_LEN + i] = prefix[i];
        // 拼接value
        for (int j = 0; j < value_len; ++j)
            out[idx * MAX_LEN + prefix_len + j] = values[idx * value_len + j];
        // 结尾符
        out[idx * MAX_LEN + prefix_len + value_len] = '\0';
    }
}
