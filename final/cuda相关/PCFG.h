#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include "mpi.h"
#include <sstream>
#include <cstring>

// #include <chrono>   
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    segment() : type(0), length(0), total_freq(0) {}
    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();

    /*
    // 序列化为字符串
    std::string serialize() const {
        std::ostringstream oss;
        oss << type << " " << length << " " << total_freq << " ";
        oss << ordered_values.size() << " ";
        for (const auto& s : ordered_values) {
            oss << s.size() << " " << s;
        }
        oss << " " << ordered_freqs.size() << " ";
        for (int f : ordered_freqs) oss << f << " ";
        return oss.str();
    }

    // 反序列化
    void deserialize(std::istringstream& iss) {
        size_t n, m;
        iss >> type >> length >> total_freq >> n;
        ordered_values.clear();
        for (size_t i = 0; i < n; ++i) {
            size_t len;
            iss >> len;
            std::string s(len, ' ');
            iss.read(&s[0], len);
            ordered_values.push_back(s);
        }
        iss >> m;
        ordered_freqs.clear();
        for (size_t i = 0; i < m; ++i) {
            int f;
            iss >> f;
            ordered_freqs.push_back(f);
        }
    }
    */    
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();
    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;

    /*
    std::string serialize() const {
        std::ostringstream oss;
        oss << content.size() << " ";
        for (const auto& seg : content) {
            std::string seg_str = seg.serialize();
            oss << seg_str.size() << " " << seg_str;
        }
        oss << pivot << " ";
        oss << curr_indices.size() << " ";
        for (int v : curr_indices) oss << v << " ";
        oss << max_indices.size() << " ";
        for (int v : max_indices) oss << v << " ";
        oss << preterm_prob << " " << prob << " ";
        return oss.str();
    }

    void deserialize(std::istringstream& iss) {
        size_t n, m;
        iss >> n;
        content.resize(n);
        for (size_t i = 0; i < n; ++i) {
            size_t seg_len;
            iss >> seg_len;
            std::string seg_str(seg_len, ' ');
            iss.read(&seg_str[0], seg_len);
            std::istringstream seg_iss(seg_str);
            content[i].deserialize(seg_iss);
        }
        iss >> pivot >> m;
        curr_indices.resize(m);
        for (size_t i = 0; i < m; ++i) iss >> curr_indices[i];
        iss >> m;
        max_indices.resize(m);
        for (size_t i = 0; i < m; ++i) iss >> max_indices[i];
        iss >> preterm_prob >> prob;
    }
    */
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);
    void GenerateMPI(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext() ;
    int total_guesses = 0;
    vector<string> guesses;

    void PopNextMPI();

    void PopNextBatchMPI(int batch_size,int rank,int size);

};

