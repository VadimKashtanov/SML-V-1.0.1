#include "package/insts/activation/head/activation.cuh"

void activation_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint _len=mdl->param[inst][0],			\
		 activ=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3];

	activation_use_th11<<<dim3(KERN_DIV(_len,32)),dim3(32)>>>(
		_len,
		activ,
		time,
		mdl->total,
		istart, ystart,
		use->var_d);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void activation_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint _len=mdl->param[inst][0],			\
		 activ=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],		\
		 lstart=mdl->param[inst][4];

	activation_forward_th11<<<dim3(KERN_DIV(_len,16),train->sets),dim3(16,1)>>>(
		_len,
		activ,
		time,
		istart, ystart, lstart,
		train->mdl->total, train->mdl->locds,
		train->_var, train->_locd,
		train->sets);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//-------------------------- backward ---------------------

void activation_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint _len=mdl->param[inst][0],			\
		 activ=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],		\
		 lstart=mdl->param[inst][4];

	activation_backward_th11<<<dim3(KERN_DIV(_len,16),train->sets),dim3(16,1)>>>(
		_len,
		activ,
		time,
		istart, ystart, lstart,
		mdl->total, mdl->locds,
		train->_var, train->_locd, train->_grad,
		train->sets);
	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};