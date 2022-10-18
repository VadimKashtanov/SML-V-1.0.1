#include "package/insts/inst/head/inst.cuh"

void inst_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint param0=mdl->param[inst][0],\
		 param1=mdl->param[inst][1];

	inst_use_mod0<<<dim3(KERN_DIV(Yx,32)),dim3(32)>>>(
		param0, param1,
		time,
		mdl->total,
		input_start, ystart, wstart,
		use->var_d, use->weight_d);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint param0=mdl->param[inst][0],\
		 param1=mdl->param[inst][1];

	float param1_100 = param1/100;

	inst_forward_mod0<<<dim3(KERN_DIV(param1,16),train->sets),dim3(16,1)>>>(
		param0, param1,
		time,
		input_start, ystart, wstart, locdstart,
		train->mdl->total, train->mdl->weights, train->mdl->locds,
		train->_var, train->_weight, train->_locd,
		inst*start_seed, drop_rate,
		train->sets);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//-------------------------- backward ---------------------

void dot1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint param0=mdl->param[inst][0],\
		 param1=mdl->param[inst][1];

	float param1_100 = param1/100;

	inst_backward_mod0<<<dim3(KERN_DIV(param1,16),train->sets),dim3(16,1)>>>(
		param0, param1,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};