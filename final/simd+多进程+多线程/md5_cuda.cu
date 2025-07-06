#include "md5_cuda.cuh"
#include <iostream>
#include <string>
#include <cstring>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;
// 你可以直接移植md5.h中的宏定义和辅助函数到这里

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))
#define FF(a, b, c, d, x, s, ac) { (a) += F((b),(c),(d)) + (x) + ac; (a) = ROTATELEFT((a),(s)); (a) += (b); }
#define GG(a, b, c, d, x, s, ac) { (a) += G((b),(c),(d)) + (x) + ac; (a) = ROTATELEFT((a),(s)); (a) += (b); }
#define HH(a, b, c, d, x, s, ac) { (a) += H((b),(c),(d)) + (x) + ac; (a) = ROTATELEFT((a),(s)); (a) += (b); }
#define II(a, b, c, d, x, s, ac) { (a) += I((b),(c),(d)) + (x) + ac; (a) = ROTATELEFT((a),(s)); (a) += (b); }

#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

__device__ void md5_single(const char* input, int length, bit32* state) {
    // 1. Padding
    int bitLength = length * 8;
    int paddingBits = bitLength % 512;
    if (paddingBits > 448) paddingBits += 512 - (paddingBits - 448);
    else if (paddingBits < 448) paddingBits = 448 - paddingBits;
    else if (paddingBits == 448) paddingBits = 512;
    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;
    unsigned char paddedMessage[128]; // 64+64, 足够存储padding后消息（最大64字节输入）

    // 拷贝原始消息
    for (int i = 0; i < length; ++i) paddedMessage[i] = input[i];
    paddedMessage[length] = 0x80;
    for (int i = length + 1; i < length + paddingBytes; ++i) paddedMessage[i] = 0;
    // 添加消息长度（64位小端）
    for (int i = 0; i < 8; ++i)
        paddedMessage[length + paddingBytes + i] = ((unsigned long long)length * 8 >> (i * 8)) & 0xFF;

    int n_blocks = paddedLength / 64;
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    for (int i = 0; i < n_blocks; ++i) {
        bit32 x[16];
        for (int j = 0; j < 16; ++j) {
            x[j] = (paddedMessage[i*64 + 4*j]) |
                   (paddedMessage[i*64 + 4*j + 1] << 8) |
                   (paddedMessage[i*64 + 4*j + 2] << 16) |
                   (paddedMessage[i*64 + 4*j + 3] << 24);
        }
        bit32 a = state[0], b = state[1], c = state[2], d = state[3];
        // Round 1
        FF(a, b, c, d, x[0], s11, 0xd76aa478);  FF(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF(c, d, a, b, x[2], s13, 0x242070db);  FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf);  FF(d, a, b, c, x[5], s12, 0x4787c62a);
        FF(c, d, a, b, x[6], s13, 0xa8304613);  FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8);  FF(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF(c, d, a, b, x[10], s13, 0xffff5bb1); FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122); FF(d, a, b, c, x[13], s12, 0xfd987193);
        FF(c, d, a, b, x[14], s13, 0xa679438e); FF(b, c, d, a, x[15], s14, 0x49b40821);
        // Round 2
        GG(a, b, c, d, x[1], s21, 0xf61e2562);  GG(d, a, b, c, x[6], s22, 0xc040b340);
        GG(c, d, a, b, x[11], s23, 0x265e5a51); GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d);  GG(d, a, b, c, x[10], s22, 0x2441453);
        GG(c, d, a, b, x[15], s23, 0xd8a1e681); GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6);  GG(d, a, b, c, x[14], s22, 0xc33707d6);
        GG(c, d, a, b, x[3], s23, 0xf4d50d87);  GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905); GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG(c, d, a, b, x[7], s23, 0x676f02d9);  GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);
        // Round 3
        HH(a, b, c, d, x[5], s31, 0xfffa3942);  HH(d, a, b, c, x[8], s32, 0x8771f681);
        HH(c, d, a, b, x[11], s33, 0x6d9d6122); HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44);  HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH(c, d, a, b, x[7], s33, 0xf6bb4b60);  HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6); HH(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH(c, d, a, b, x[3], s33, 0xd4ef3085);  HH(b, c, d, a, x[6], s34, 0x4881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039);  HH(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH(c, d, a, b, x[15], s33, 0x1fa27cf8); HH(b, c, d, a, x[2], s34, 0xc4ac5665);
        // Round 4
        II(a, b, c, d, x[0], s41, 0xf4292244);  II(d, a, b, c, x[7], s42, 0x432aff97);
        II(c, d, a, b, x[14], s43, 0xab9423a7); II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3); II(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II(c, d, a, b, x[10], s43, 0xffeff47d); II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f);  II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II(c, d, a, b, x[6], s43, 0xa3014314);  II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82);  II(d, a, b, c, x[11], s42, 0xbd3af235);
        II(c, d, a, b, x[2], s43, 0x2ad7d2bb);  II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }
    // 输出顺序转换
    for (int i = 0; i < 4; ++i) {
        unsigned int v = state[i];
        state[i] = ((v & 0xff) << 24) | ((v & 0xff00) << 8) | ((v & 0xff0000) >> 8) | ((v & 0xff000000) >> 24);
    }
}

__global__ void md5_batch_kernel(const char* inputs, const int* lengths, int max_len, int count, bit32* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    const char* str = inputs + idx * max_len;
    int len = lengths[idx];
    bit32 result[4];
    md5_single(str, len, result);
    for (int i = 0; i < 4; ++i)
        out[idx * 4 + i] = result[i];
}


void md5_batch_host(const std::vector<std::string>& inputs, std::vector<bit32>& out_hash) {
    int N = inputs.size();
    int max_len = 64; // 假设最大64字节
    std::vector<char> h_inputs(N * max_len, 0);
    std::vector<int> h_lengths(N, 0);
    for (int i = 0; i < N; ++i) {
        memcpy(&h_inputs[i * max_len], inputs[i].c_str(), inputs[i].size());
        h_lengths[i] = inputs[i].size();
    }
    char *d_inputs;
    int *d_lengths;
    bit32 *d_out;
    cudaMalloc(&d_inputs, N * max_len);
    cudaMalloc(&d_lengths, N * sizeof(int));
    cudaMalloc(&d_out, N * 4 * sizeof(bit32));
    cudaMemcpy(d_inputs, h_inputs.data(), N * max_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    md5_batch_kernel<<<blocks, threads>>>(d_inputs, d_lengths, max_len, N, d_out);

    out_hash.resize(N * 4);
    cudaMemcpy(out_hash.data(), d_out, N * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_lengths);
    cudaFree(d_out);
}
