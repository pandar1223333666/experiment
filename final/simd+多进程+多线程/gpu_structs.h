#pragma once
#define MAX_SEG 3

struct PT_GPU {
    int seg_num;
    int curr_indices[MAX_SEG];
    int max_indices[MAX_SEG];
    int types[MAX_SEG];
    int type_indices[MAX_SEG];
};

struct Segment_GPU {
    int value_num;
    int value_offset;
};