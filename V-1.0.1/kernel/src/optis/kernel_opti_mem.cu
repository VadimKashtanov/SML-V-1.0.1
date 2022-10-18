#include "kernel/head/optis.cuh"

/*		-------------    Build  ------------ */
Opti_t * opti_mk(Train_t * train, uint score_algo, uint opti_algo) {
	if (score_algo >= SCORES)
		ERR("Score number %i doesn't exists. Max is %i", score_algo, SCORES - 1)
	if (opti_algo >= OPTIS)
		ERR("Opti number %i doesn't exists. Max is %i", opti_algo, OPTIS - 1)

	Opti_t * ret = (Opti_t*)malloc(sizeof(Opti_t));
	
	ret->train = train;

	//	Cpu ram arrays
	ret->set_score = (float*)malloc(sizeof(float) * train->sets);
	ret->set_rank = (uint*)malloc(sizeof(uint) * train->sets);

	ret->podium = (uint*)malloc(sizeof(uint) * train->sets);

	//	Gpu vram arrays
	SAFE_CUDA(cudaMalloc((void**)&ret->set_score_d, sizeof(float) * train->sets));
	SAFE_CUDA(cudaMalloc((void**)&ret->set_rank_d, sizeof(uint) * train->sets));

	//	Algorithms
	ret->score_algo = score_algo;
	ret->opti_algo = opti_algo;

	ret->score_space = OPTI_SCORE_SPACE_MK_ARRAY[score_algo](ret);
	ret->opti_space = OPTI_OPTI_SPACE_MK_ARRAY[opti_algo](ret);

	return ret;
};

/*		-------------    Free structure  ------------ */
void opti_free(Opti_t * opti) {
	free(opti->set_score);
	free(opti->set_rank);
	free(opti->podium);

	SAFE_CUDA(cudaFree(opti->set_score_d));
	SAFE_CUDA(cudaFree(opti->set_rank_d));

	OPTI_FREE_SCORE_ARRAY[opti->score_algo](opti);
	OPTI_FREE_OPTI_ARRAY[opti->opti_algo](opti);

	free(opti);
};