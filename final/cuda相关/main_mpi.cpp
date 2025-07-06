#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>             

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash  = 0;   // 用于 MD5 哈希的时间
    double time_guess = 0;   // 哈希 + 猜测的总时长
    double time_train = 0;   // 模型训练的总时长

    PriorityQueue q;

    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train   = system_clock::now();
    time_train = duration<double>(end_train - start_train).count();

    q.init();

    long long curr_num = 0;             // 累积未哈希条数
    long long history  = 0;             // 已经哈希过的条数
    long long total_generated = 0;      // 所有进程总计生成条数
    long long last_print = 0;

    auto start = system_clock::now();

    const long long generate_n = 10000000;  
    const long long hash_chunk = 1000000; 

    while (!q.priority.empty())
    {
        std::size_t before = q.guesses.size();
        q.PopNextMPI();         
        std::size_t local_added = q.guesses.size() - before;

        long long global_added = 0;
        MPI_Allreduce(&local_added, &global_added, 1,
                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        total_generated += global_added;
        curr_num        += global_added;

        if (rank == 0 && total_generated - last_print >= 100000) {
            cout << "Guesses generated: " << total_generated << endl;
            last_print = total_generated;
        }

        if (total_generated >= generate_n)
        {
            auto end = system_clock::now();
            time_guess = duration<double>(end - start).count();

            if (rank == 0) {
                cout << "Guess time:" << time_guess - time_hash << " seconds\n";
                cout << "Hash  time:" << time_hash             << " seconds\n";
                cout << "Train time:" << time_train            << " seconds\n";
            }
            break;
        }

        if (curr_num >= hash_chunk)
        {
            auto start_hash = system_clock::now();

            bit32 state[4];
            for (const string &pw : q.guesses)
                MD5Hash(pw, state);     

            auto end_hash = system_clock::now();
            time_hash += duration<double>(end_hash - start_hash).count();

            history  += curr_num;
            curr_num  = 0;
            q.guesses.clear();    
        }
    }

    MPI_Finalize();
    return 0;
}
