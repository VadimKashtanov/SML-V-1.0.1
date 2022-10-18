#pragma once

#include "kernel/head/train.cuh"

typedef struct optimizer_and_score {
	//	Train_t of model
	Train_t * train;

	//	Ranking
	float * set_score;		//	score of i'th set
	float * set_score_d;
	uint * set_rank;		//set_rank[i] give the place on podium of i'th set
	uint * set_rank_d;
	uint * podium;

	//	Algorithms
	uint score_algo, opti_algo;
	void * score_space, * opti_space;
} Opti_t;

//		Mem
Opti_t * opti_mk(Train_t * train, uint score_algo, uint opti_algo);

void opti_opti_set_one_arg(Opti_t * opti, char * argname, char * value);
void opti_score_set_one_arg(Opti_t * opti, char * argname, char * value);

void opti_opti_load_consts(Opti_t * opti, FILE * fp);
void opti_score_load_consts(Opti_t * opti, FILE * fp);

//		Controle
void opti_loss(Opti_t * opti);
void opti_dloss(Opti_t * opti);
void opti_opti(Opti_t * opti);

//		Plum
void opti_print_scores(Opti_t * opti);
void opti_compare_scores(Opti_t * opti, float * with_this);
void opti_print_rank(Opti_t * opti);
void opti_print_podium(Opti_t * opti);

//		Free
void opti_free(Opti_t * opti);

////	Arrays that all fonctions above uses
extern uint OPTI_MIN_ECHOPES[OPTIS];

// =======	Build Score & Optimizer space =======
extern void* (*OPTI_SCORE_SPACE_MK_ARRAY[SCORES])(Opti_t * opti);
extern void* (*OPTI_OPTI_SPACE_MK_ARRAY[OPTIS])(Opti_t * opti);

//	=============	CONSTS  ===============
extern void (*OPTI_SCORE_SET_ONE_ARG_ARRAY[SCORES])(Opti_t * opti, char * name, char * value);
extern void (*OPTI_OPTI_SET_ONE_ARG_ARRAY[OPTIS])(Opti_t * opti, char * name, char * value);

extern const uint OPTI_CONST_AMOUNT[OPTIS];
extern const uint SCORE_CONST_AMOUNT[SCORES];

extern const char ** OPTI_CONST_ARRAY[OPTIS];
extern const char ** SCORE_CONST_ARRAY[SCORES];

// ============= Dloss of Train_t ==============
extern void (*OPTI_SCORES_DLOSS_ARRAY[SCORES])(Opti_t * opti);

// ============= Optimize weights of Train_t ===========
extern void (*OPTI_OPTIMIZE_ARRAY[OPTIS])(Opti_t * opti);

// ============ Compute Score and rank it =========
extern void (*OPTI_COMPUTE_LOSS_ARRAY[SCORES])(Opti_t * opti);

// ============== Free the structure ===========
extern void (*OPTI_FREE_SCORE_ARRAY[SCORES])(Opti_t * opti);
extern void (*OPTI_FREE_OPTI_ARRAY[OPTIS])(Opti_t * opti);