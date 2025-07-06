#pragma once

__global__ void generate_guesses_kernel(const char* values, int value_len, int value_count, char* out);
__global__ void generate_guesses_with_prefix_kernel(
    const char* prefix, int prefix_len,
    const char* values, int value_len,
    int value_count, char* out);