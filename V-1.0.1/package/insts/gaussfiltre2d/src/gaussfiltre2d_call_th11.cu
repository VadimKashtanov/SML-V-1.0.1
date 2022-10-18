#include "package/insts/gaussfiltre2d/head/gaussfiltre2d.cuh"

void gaussfiltre2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint X=mdl->param[inst][0],		\
		 Y=mdl->param[inst][1],		\
		 istart=mdl->param[inst][2],\
		 ystart=mdl->param[inst][3],\
		 wstart=mdl->param[inst][4];

	gaussfiltre2d_use_th11<<<dim3(KERN_DIV(X,16), KERN_DIV(Y,16)),dim3(16,16)>>>(
		X, Y,
		time,
		mdl->total,
		istart, ystart, wstart,
		use->var_d, use->weight_d);
}

//======================== Train_t =======================

//-------------------------- forward ---------------------

void gaussfiltre2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 lstart=mdl->param[inst][5];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint lsize = mdl->locds;

	uint sets = train->sets;

	gaussfiltre2d_forward_th11<<<dim3(KERN_DIV(X,16), KERN_DIV(Y,16),sets),dim3(16,16,1)>>>(
		X, Y,
		time,
		istart, ystart, wstart, lstart,
		total, wsize, lsize,
		train->_var, train->_weight, train->_locd,
		train->sets);
}

//-------------------------- backward ---------------------

void gaussfiltre2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 lstart=mdl->param[inst][5];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint lsize = mdl->locds;

	uint sets = train->sets;

	gaussfiltre2d_backward_th11<<<dim3(KERN_DIV(X,16), KERN_DIV(Y,16), sets),dim3(16,16,1)>>>(
		X, Y,
		time,
		istart, ystart, wstart, lstart,
		total, wsize, lsize,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		train->sets);
}