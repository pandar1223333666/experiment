#include "PCFG.h"
#include <cstring>
#include <sstream>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
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
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
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

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
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


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
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
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
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
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

#include <sstream>

// segment 类的序列化方法
string segment::serialize() 
{
    stringstream ss;
    ss << type << " " << length << " " << total_freq << " ";
    ss << ordered_values.size() << " ";
    for (const auto& value : ordered_values)
    {
        ss << value << " ";
    }
    ss << ordered_freqs.size() << " ";
    for (const auto& freq : ordered_freqs)
    {
        ss << freq << " ";
    }
    ss << values.size() << " ";
    for (const auto& pair : values)
    {
        ss << pair.first << " " << pair.second << " ";
    }
    ss << freqs.size() << " ";
    for (const auto& pair : freqs)
    {
        ss << pair.first << " " << pair.second << " ";
    }
    return ss.str();
}

// segment 类的反序列化方法
void segment::deserialize(string& data)
{
    stringstream ss(data);
    ss >> type >> length >> total_freq;
    size_t ordered_values_size;
    ss >> ordered_values_size;
    ordered_values.clear();
    for (size_t i = 0; i < ordered_values_size; ++i)
    {
        string value;
        ss >> value;
        ordered_values.push_back(value);
    }
    size_t ordered_freqs_size;
    ss >> ordered_freqs_size;
    ordered_freqs.clear();
    for (size_t i = 0; i < ordered_freqs_size; ++i)
    {
        int freq;
        ss >> freq;
        ordered_freqs.push_back(freq);
    }
    size_t values_size;
    ss >> values_size;
    values.clear();
    for (size_t i = 0; i < values_size; ++i)
    {
        string key;
        int val;
        ss >> key >> val;
        values[key] = val;
    }
    size_t freqs_size;
    ss >> freqs_size;
    freqs.clear();
    for (size_t i = 0; i < freqs_size; ++i)
    {
        int key;
        int val;
        ss >> key >> val;
        freqs[key] = val;
    }
}

// PT 类的序列化方法
string PT::serialize() 
{
    stringstream ss;
    ss << pivot << " " << preterm_prob << " " << prob << " ";
    ss << content.size() << " ";
    for (const auto& seg : content)
    {
        ss << seg.serialize() << " ";
    }
    ss << curr_indices.size() << " ";
    for (const auto& idx : curr_indices)
    {
        ss << idx << " ";
    }
    ss << max_indices.size() << " ";
    for (const auto& idx : max_indices)
    {
        ss << idx << " ";
    }
    return ss.str();
}

// PT 类的反序列化方法
void PT::deserialize(string& data)
{
    stringstream ss(data);
    ss >> pivot >> preterm_prob >> prob;
    size_t content_size;
    ss >> content_size;
    content.clear();
    for (size_t i = 0; i < content_size; ++i)
    {
        string seg_data;
        size_t seg_size;
        ss >> seg_size;
        for (size_t j = 0; j < seg_size; ++j)
        {
            string token;
            ss >> token;
            seg_data += token + " ";
        }
        segment seg;
        seg.deserialize(seg_data);
        content.push_back(seg);
    }
    size_t curr_indices_size;
    ss >> curr_indices_size;
    curr_indices.clear();
    for (size_t i = 0; i < curr_indices_size; ++i)
    {
        int idx;
        ss >> idx;
        curr_indices.push_back(idx);
    }
    size_t max_indices_size;
    ss >> max_indices_size;
    max_indices.clear();
    for (size_t i = 0; i < max_indices_size; ++i)
    {
        int idx;
        ss >> idx;
        max_indices.push_back(idx);
    }
}

void PriorityQueue::PopNextBatchMPI(int batch_size, int rank, int size)
{
    std::vector<PT> local_batch;
    if (rank == 0)
    {
        // 主进程：将优先队列中的PT分发给各个从进程
        std::vector<PT> global_batch;
        for (int i = 0; i < batch_size && !priority.empty(); ++i)
        {
            global_batch.push_back(priority.front());
            priority.erase(priority.begin());
        }

        int local_batch_size = global_batch.size() / size;
        int remainder = global_batch.size() % size;

        for (int i = 1; i < size; ++i)
        {
            int start = i * local_batch_size + std::min(i, remainder);
            int end = start + local_batch_size + (i < remainder ? 1 : 0);
            int send_size = end - start;
            MPI_Send(&send_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            for (int j = start; j < end; ++j)
            {
                std::string pt_data = global_batch[j].serialize();
                int data_size = pt_data.size();
                MPI_Send(&data_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(pt_data.c_str(), data_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }

        // 主进程处理自己的任务
        int start = 0;
        int end = local_batch_size + (0 < remainder ? 1 : 0);
        for (int j = start; j < end; ++j)
        {
            local_batch.push_back(global_batch[j]);
        }
    }
    else
    {
        // 从进程：接收主进程分发的PT
        int recv_size;
        MPI_Recv(&recv_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < recv_size; ++i)
        {
            int data_size;
            MPI_Recv(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string pt_data(data_size, '\0');
            MPI_Recv(&pt_data[0], data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            PT pt;
            pt.deserialize(pt_data);
            local_batch.push_back(pt);
        }
    }

    // 每个进程处理分配到的PT
    std::vector<PT> new_pts;
    for (const auto& pt : local_batch)
    {
        Generate(pt);
        std::vector<PT> pt_new_pts = pt.NewPTs();
        for (PT& new_pt : pt_new_pts)
        {
            CalProb(new_pt);
            new_pts.push_back(new_pt);
        }
    }

    if (rank != 0)
    {
        // 从进程：将生成的新PT发送回主进程
        int send_size = new_pts.size();
        MPI_Send(&send_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (const auto& new_pt : new_pts)
        {
            std::string pt_data = new_pt.serialize();
            int data_size = pt_data.size();
            MPI_Send(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(pt_data.c_str(), data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // 主进程：收集从进程发送的新PT
        for (int i = 1; i < size; ++i)
        {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < recv_size; ++j)
            {
                int data_size;
                MPI_Recv(&data_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string pt_data(data_size, '\0');
                MPI_Recv(&pt_data[0], data_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                PT new_pt;
                new_pt.deserialize(pt_data);
                new_pts.push_back(new_pt);
            }
        }

        // 将所有新的PT放回优先队列
        for (const auto& new_pt : new_pts)
        {
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); ++iter)
            {
                if (new_pt.prob <= iter->prob && (iter + 1 == priority.end() || new_pt.prob > (iter + 1)->prob))
                {
                    priority.insert(iter + 1, new_pt);
                    inserted = true;
                    break;
                }
            }
            if (!inserted)
            {
                if (priority.empty() || new_pt.prob > priority.front().prob)
                {
                    priority.insert(priority.begin(), new_pt);
                }
                else
                {
                    priority.push_back(new_pt);
                }
            }
        }
    }
}



