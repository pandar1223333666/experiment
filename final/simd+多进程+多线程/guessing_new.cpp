#include "PCFG.h"
using namespace std;

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


void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

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

        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
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
        

        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

/*
void PriorityQueue::PopBatch(int batch_size)
{
    int n = std::min(batch_size, (int)priority.size());
    std::vector<PT> batch;
    for (int i = 0; i < n; ++i) {
        batch.push_back(priority.front());
        priority.erase(priority.begin());
    }

    BatchGenerateGPU(batch);

    for (PT& pt : batch) {
        std::vector<PT> new_pts = pt.NewPTs();
        for (PT new_pt : new_pts) {
            CalProb(new_pt);
            auto iter = priority.begin();
            while (iter != priority.end() && new_pt.prob < iter->prob) ++iter;
            priority.insert(iter, new_pt);
        }
    }
}
*/

void PriorityQueue::PopBatch(int batch_size) {
    std::vector<PT> batch;
    int total_combos = 0;
    const int MAX_COMBOS = 100000; 
    
    for (int i = 0; i < batch_size && !priority.empty(); ++i) {
        PT& pt = priority.front();
        int pt_combos = 1;
        for (int j = 0; j < pt.content.size(); ++j) {
            pt_combos *= pt.max_indices[j];
        }
        
        if (total_combos + pt_combos > MAX_COMBOS) {
            break;  // 停止添加，避免溢出
        }
        
        batch.push_back(pt);
        total_combos += pt_combos;
        priority.erase(priority.begin());
    }
    
    if (!batch.empty()) {
        BatchGenerateGPU(batch);
    }
}




#include "gpu_structs.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
extern "C" void launch_batch_generate_kernel(
    PT_GPU* pts, Segment_GPU* letters, Segment_GPU* digits, Segment_GPU* symbols,
    char* output, int* output_offsets, int* pt_offsets, int batch_size, int total_combos,
    int threads, int blocks);

void PriorityQueue::BatchGenerateGPU(const vector<PT>& batch)
{
    int batch_size = batch.size();
    PT_GPU* h_pts = new PT_GPU[batch_size];

    // 1. 打包PT数据
    for (int i = 0; i < batch_size; ++i) {
        const PT& pt = batch[i];
        for (int j = 0; j < pt.content.size(); ++j) {
            h_pts[i].curr_indices[j] = pt.curr_indices[j];
            h_pts[i].max_indices[j] = pt.max_indices[j];
            h_pts[i].types[j] = pt.content[j].type;
            if (pt.content[j].type == 1)
                h_pts[i].type_indices[j] = m.FindLetter(pt.content[j]);
            else if (pt.content[j].type == 2)
                h_pts[i].type_indices[j] = m.FindDigit(pt.content[j]);
            else if (pt.content[j].type == 3)
                h_pts[i].type_indices[j] = m.FindSymbol(pt.content[j]);
        }
        h_pts[i].seg_num = pt.content.size();
    }

    // 2. 打包segment数据
    int letters_num = m.letters.size();
    int digits_num = m.digits.size();
    int symbols_num = m.symbols.size();

    vector<char> all_values;
    vector<Segment_GPU> h_letters(letters_num);
    vector<Segment_GPU> h_digits(digits_num);
    vector<Segment_GPU> h_symbols(symbols_num);

    int offset = 0;

    for (int i = 0; i < letters_num; ++i) {
        h_letters[i].value_num = m.letters[i].ordered_values.size();
        h_letters[i].value_offset = offset;
        for (const auto& val : m.letters[i].ordered_values) {
            all_values.insert(all_values.end(), val.begin(), val.end());
            all_values.push_back('\0');
            offset += val.size() + 1;
        }
    }

    for (int i = 0; i < digits_num; ++i) {
        h_digits[i].value_num = m.digits[i].ordered_values.size();
        h_digits[i].value_offset = offset;
        for (const auto& val : m.digits[i].ordered_values) {
            all_values.insert(all_values.end(), val.begin(), val.end());
            all_values.push_back('\0');
            offset += val.size() + 1;
        }
    }

    for (int i = 0; i < symbols_num; ++i) {
        h_symbols[i].value_num = m.symbols[i].ordered_values.size();
        h_symbols[i].value_offset = offset;
        for (const auto& val : m.symbols[i].ordered_values) {
            all_values.insert(all_values.end(), val.begin(), val.end());
            all_values.push_back('\0');
            offset += val.size() + 1;
        }
    }

    // 计算每个PT的组合数和全局偏移
    std::vector<int> pt_offsets(batch_size + 1, 0);
    int total_combos = 0;
    for (int i = 0; i < batch_size; ++i) {
        int combos = 1;
        for (int j = 0; j < h_pts[i].seg_num; ++j) {
            combos *= h_pts[i].max_indices[j];
        }
        pt_offsets[i] = total_combos;
        total_combos += combos;
    }
    pt_offsets[batch_size] = total_combos;

    const int MAX_COMBOS_PER_BATCH = 1000000; // 可根据实际内存调整
    if (total_combos > MAX_COMBOS_PER_BATCH) {
        cout << "[ERROR] total_combos 太大，跳过本批: " << total_combos << endl;
        delete[] h_pts;
        return;
    }

    // 4. 拷贝到GPU
    PT_GPU* d_pts;
    Segment_GPU* d_letters;
    Segment_GPU* d_digits;
    Segment_GPU* d_symbols;
    int* d_pt_offsets;
    char* d_all_values;

    cudaMalloc(&d_pts, batch_size * sizeof(PT_GPU));
    cudaMemcpy(d_pts, h_pts, batch_size * sizeof(PT_GPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_letters, letters_num * sizeof(Segment_GPU));
    cudaMemcpy(d_letters, h_letters.data(), letters_num * sizeof(Segment_GPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_digits, digits_num * sizeof(Segment_GPU));
    cudaMemcpy(d_digits, h_digits.data(), digits_num * sizeof(Segment_GPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_symbols, symbols_num * sizeof(Segment_GPU));
    cudaMemcpy(d_symbols, h_symbols.data(), symbols_num * sizeof(Segment_GPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pt_offsets, (batch_size + 1) * sizeof(int));
    cudaMemcpy(d_pt_offsets, pt_offsets.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_all_values, all_values.size());
    cudaMemcpy(d_all_values, all_values.data(), all_values.size(), cudaMemcpyHostToDevice);

    // 5. 分配输出空间和offsets
    int max_guess_len = 128;
    char* d_output;
    int* d_output_offsets;
    std::vector<int> output_offsets(total_combos);
    for (int i = 0; i < total_combos; ++i) output_offsets[i] = i * max_guess_len;
    cudaMalloc(&d_output, total_combos * max_guess_len);
    cudaMalloc(&d_output_offsets, total_combos * sizeof(int));
    cudaMemcpy(d_output_offsets, output_offsets.data(), total_combos * sizeof(int), cudaMemcpyHostToDevice);
   
    // 启动kernel
    int threads = 256;
    int blocks = (total_combos + threads - 1) / threads;
    
    launch_batch_generate_kernel(d_pts, d_letters, d_digits, d_symbols, d_output, d_output_offsets, d_pt_offsets,
        batch_size, total_combos, threads, blocks);

    //  拷贝结果回主机
    char* h_output = new char[total_combos * max_guess_len];
    cudaMemcpy(h_output, d_output, total_combos * max_guess_len, cudaMemcpyDeviceToHost);

    // 合并到guesses
    for (int i = 0; i < total_combos; ++i) {
        guesses.emplace_back(h_output + i * max_guess_len);
        total_guesses += 1;
    }

    // 9. 释放内存
    delete[] h_pts;
    delete[] h_output;
    cudaFree(d_pts);
    cudaFree(d_letters);
    cudaFree(d_digits);
    cudaFree(d_symbols);
    cudaFree(d_output);
    cudaFree(d_output_offsets);
    cudaFree(d_pt_offsets);
    cudaFree(d_all_values);
}

