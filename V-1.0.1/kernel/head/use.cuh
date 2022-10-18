#pragma once

#include "kernel/head/mdl.cuh"
#include "kernel/head/data.cuh"

typedef struct gpu_model_forward {
	Mdl_t * mdl;
	Data_t * data;

	//Weight of Model stored on nvidia vram
	float * weight_d;

	//Variable space of mdl instructions on nvidia's vram
	//times == data->lines;
	float * var_d;
} Use_t;

//	Mem
Use_t* use_mk(Mdl_t * mdl, Data_t * data);

//	Controle
void use_set_input(Use_t * use);	//batch size == use->times == data->lines
void use_forward(Use_t * use);

//	Free
void use_free(Use_t * use);

//	Plum
void use_print_weights(Use_t * use);
void use_print_vars(Use_t * use);
//
void use_print_all(Use_t * use);
//
void use_compare_weights(Use_t * use, float * with_this);
void use_compare_vars(Use_t * use, float * with_this);

//	Use instructions for nvidia card forward
typedef void (*use_f)(Use_t* use, uint inst, uint time);
extern use_f INST_USE[INSTS];