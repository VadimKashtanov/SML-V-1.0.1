#pragma once

#include "optis.cuh"

Data_t * load_test_data(FILE * fp);
float * load_float_array(uint len, FILE * fp);
void read_123(FILE * fp);

bool test_package_compare_cpu_and_gpu(float * cpu0, float * gpu_d, uint count);
bool test_package_compare_cpu_and_cpu(float * cpu0, float * cpu1, uint count);

void test_mdl(bool print_all, FILE * fp);
void test_score(bool print_all, FILE * fp);
void test_opti(bool print_all, FILE * fp);