#include "kernel/head/train.cuh"

Train_t* mk_train(Mdl_t * mdl, Data_t * data, uint sets)
{
	Train_t * ret = (Train_t*)malloc(sizeof(Train_t));

	ret->mdl = mdl;
	ret->data = data;
	ret->sets = sets;

	uint lines = data->lines;

	SAFE_CUDA(cudaMalloc((void**)&ret->_weight, sizeof(float) * (mdl->weights*sets)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_var, sizeof(float) * (mdl->total*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_locd, sizeof(float) * (mdl->locds*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_grad, sizeof(float) * (mdl->total*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_meand, sizeof(float) * (mdl->weights*sets)));

	return ret;
};

void train_free(Train_t * train) {
	SAFE_CUDA(cudaFree(train->_weight));
	SAFE_CUDA(cudaFree(train->_var));
	SAFE_CUDA(cudaFree(train->_locd));
	SAFE_CUDA(cudaFree(train->_grad));
	SAFE_CUDA(cudaFree(train->_meand));

	free(train);
};

