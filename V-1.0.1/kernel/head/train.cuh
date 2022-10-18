#pragma once

#include "kernel/head/data.cuh"
#include "kernel/head/mdl.cuh"

#include "kernel/head/use.cuh"
#include "kernel/head/cpu.cuh"

//	Locd is very usefull for max finding. Beacause, otherwise you will have to re-compute all the forward

typedef struct train_model {
	//	Model and sets trainned
	Mdl_t * mdl;
	uint sets;
	
	//	Data
	Data_t * data;

	//	Vram arrays
	float * _weight;	//_weight[sets ][wsize]
	float * _var;		//   _var[times][sets ][vsize]
	float * _locd;		//  _locd[times][sets ][lsize]
	float * _grad;		//  _grad[times][sets ][vsize]
	float * _meand;		// _meand[sets ][wsize]
} Train_t;

//	Mem
Train_t* mk_train(Mdl_t * mdl, Data_t * data, uint sets);
void train_random_weights(Train_t * train, uint rnd_seed);
void train_random_weights_from_mdl(Train_t * train, uint rnd_seed);
void train_cpy_ws_to_mdl(Train_t * train, uint set);
Train_t * extract_to_new_train(Train_t * old, uint amount, uint * set_id);	//extract set[0], set[4], set[2], set[1] and set[23]  to the new train and in this order

//pas obliatoire
//void train_snapshot(Train_t * train);	//////////////////////////!!!!!!!!!!!!!!!!!!!!!!!

//	Controle
void train_set_input(Train_t * train);
void train_null_grad_meand(Train_t * train);
void train_forward(Train_t * train, uint start_seed);
void train_backward(Train_t * train, uint start_seed);

//	Free
void train_free(Train_t * train);

//	Plum
void train_print_weights(Train_t * train);
void train_print_vars(Train_t * train);
void train_print_locds(Train_t * train);
void train_print_grads(Train_t * train);
void train_print_meands(Train_t * train);
//
void train_print_all(Train_t * train);
//
void train_compare_weights(Train_t * train, float * with_this);	//all are cpu RAM alloced arrays
void train_compare_vars(Train_t * train, float * with_this);
void train_compare_locds(Train_t * train, float * with_this);
void train_compare_grads(Train_t * train, float * with_this);
void train_compare_meands(Train_t * train, float * with_this);

typedef void (*train_f)(Train_t* train, uint inst, uint time, uint start_seed);
extern train_f INST_FORWARD[INSTS];
extern train_f INST_BACKWARD[INSTS];