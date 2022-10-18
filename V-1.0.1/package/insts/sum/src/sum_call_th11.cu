#include "package/insts/sum/head/sum.cuh"

void sum_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint size   = mdl->param[inst][0],	\
		 items  = mdl->param[inst][1],	\
		 istart = mdl->param[inst][2],	\
		 ystart = mdl->param[inst][3];

	sum_use_th11<<<dim3(KERN_DIV(size,32)),dim3(32)>>>(
		size, items,
		time,
		mdl->total,
		istart, ystart,
		use->var_d);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void sum_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint size   = mdl->param[inst][0],	\
		 items  = mdl->param[inst][1],	\
		 istart = mdl->param[inst][2],	\
		 ystart = mdl->param[inst][3];

	sum_forward_th11<<<dim3(KERN_DIV(size,32), train->sets),dim3(32, 1)>>>(
		size, items,
		time,
		mdl->total, mdl->locds,
		istart, ystart,
		train->sets,
		train->_var);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//-------------------------- backward ---------------------

void sum_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint size   = mdl->param[inst][0],	\
		 items  = mdl->param[inst][1],	\
		 istart = mdl->param[inst][2],	\
		 ystart = mdl->param[inst][3];

	sum_backward_th11<<<dim3(KERN_DIV(size,32), train->sets),dim3(32, 1)>>>(
		size, items,
		time,
		mdl->total, mdl->locds,
		istart, ystart,
		train->sets,
		train->_var, train->_grad);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};