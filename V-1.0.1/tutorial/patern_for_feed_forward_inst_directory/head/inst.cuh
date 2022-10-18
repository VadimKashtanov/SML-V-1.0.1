#pragma once

#include "kernel/head/train.cuh"

#include "inst_th11.cuh"

void inst_check(uint * param);

//======================= Cpu_t forward ===========================

void inst_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void inst_use_call_mode_mod0(Use_t * use, uint inst, uint time);

void inst_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void inst_forward_call_mode_mod0(Train_t * train, uint inst, uint time, uint start_seed);

void inst_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void inst_backward_call_mode_mod0(Train_t * train, uint inst, uint time, uint start_seed);

void inst_backward(Train_t * train, uint inst, uint time, uint start_seed);