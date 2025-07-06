#include "PCFG.h"
using namespace std;

// nvcc -O2 -DUSE_CUDA main.cpp train.cpp guessing.cu md5.cpp guessing_cuda.cu -o test.exe



void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;

        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}



/*
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "guessing_cuda.cuh"
#include <cstring>
#define MAX_LEN 64

static char *d_values = nullptr, *d_out = nullptr, *d_prefix = nullptr;
static size_t last_values_size = 0, last_out_size = 0, last_prefix_size = 0;
#endif
*/

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

/*
#ifdef USE_CUDA
    static cudaStream_t stream = nullptr;
    if (!stream) cudaStreamCreate(&stream);
#endif
*/

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

/*
#ifdef USE_CUDA
        int value_count = pt.max_indices[0];
        int value_len = 0;
        if (!a->ordered_values.empty())
            value_len = a->ordered_values[0].size();

        // 打包所有value为连续内存
        std::vector<char> h_values(value_count * value_len);
        for (int i = 0; i < value_count; ++i)
            memcpy(&h_values[i * value_len], a->ordered_values[i].c_str(), value_len);

        // 内存复用：只在需要时分配或扩容
        size_t values_size = value_count * value_len;
        size_t out_size = value_count * MAX_LEN;
        if (!d_values || last_values_size < values_size) {
            if (d_values) cudaFree(d_values);
            cudaMalloc(&d_values, values_size);
            last_values_size = values_size;
        }
        if (!d_out || last_out_size < out_size) {
            if (d_out) cudaFree(d_out);
            cudaMalloc(&d_out, out_size);
            last_out_size = out_size;
        }

        
        cudaMemcpyAsync(d_values, h_values.data(), values_size, cudaMemcpyHostToDevice,stream);

        // 启动kernel
        int threads = 512;
        int blocks = (value_count + threads - 1) / threads;
        //generate_guesses_kernel<<<blocks, threads>>>(d_values, value_len, value_count, d_out);
        generate_guesses_kernel<<<blocks, threads, 0, stream>>>(d_values, value_len, value_count, d_out);

        //  拷贝结果回主机
        std::vector<char> h_out(out_size);
        //cudaMemcpy(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        //  插入guesses
        for (int i = 0; i < value_count; ++i)
            guesses.emplace_back(&h_out[i * MAX_LEN]);
        total_guesses += value_count;

#else
*/
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
//#endif
    }
    else
    {
        string guess;
        int seg_idx = 0;
        
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
  
/*        
#ifdef USE_CUDA
        int value_count = pt.max_indices[pt.content.size() - 1];
        int value_len = 0;
        if (!a->ordered_values.empty())
            value_len = a->ordered_values[0].size();

        // 打包所有value为连续内存
        std::vector<char> h_values(value_count * value_len);
        for (int i = 0; i < value_count; ++i)
            memcpy(&h_values[i * value_len], a->ordered_values[i].c_str(), value_len);

        size_t prefix_size = guess.size();
        size_t values_size = value_count * value_len;
        size_t out_size = value_count * MAX_LEN;
        if (!d_prefix || last_prefix_size < prefix_size) {
            if (d_prefix) cudaFree(d_prefix);
            cudaMalloc(&d_prefix, prefix_size);
            last_prefix_size = prefix_size;
        }
        if (!d_values || last_values_size < values_size) {
            if (d_values) cudaFree(d_values);
            cudaMalloc(&d_values, values_size);
            last_values_size = values_size;
        }
        if (!d_out || last_out_size < out_size) {
            if (d_out) cudaFree(d_out);
            cudaMalloc(&d_out, out_size);
            last_out_size = out_size;
        }
        
        cudaMemcpyAsync(d_values, h_values.data(), values_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_prefix, guess.data(), prefix_size, cudaMemcpyHostToDevice, stream);

        // 启动kernel
        int threads = 256;
        int blocks = (value_count + threads - 1) / threads;
        //generate_guesses_with_prefix_kernel<<<blocks, threads>>>(
        //    d_prefix, prefix_size, d_values, value_len, value_count, d_out);

        generate_guesses_with_prefix_kernel<<<blocks, threads, 0, stream>>>(
            d_prefix, prefix_size, d_values, value_len, value_count, d_out);


        // 拷贝结果回主机
        std::vector<char> h_out(out_size);
        //cudaMemcpy(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        // 插入guesses
        for (int i = 0; i < value_count; ++i)
            guesses.emplace_back(&h_out[i * MAX_LEN]);
        total_guesses += value_count;

#else
*/
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
//#endif
    }
}