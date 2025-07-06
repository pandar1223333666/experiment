#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>
typedef unsigned int bit32;

void md5_batch_host(const std::vector<std::string>& inputs, std::vector<bit32>& out_hash);

__global__ void md5_batch_kernel(const char* inputs, const int* lengths, int max_len, int count, bit32* out);