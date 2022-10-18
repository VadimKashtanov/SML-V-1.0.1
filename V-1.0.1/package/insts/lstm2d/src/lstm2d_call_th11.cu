#include "package/insts/lstm2d/head/lstm2d.cuh"

// =======================================================
// ===================== Use_t ===========================
// =======================================================

void lstm2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5];

	lstm2d_use_th11<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16)),dim3(16,16)>>>(
		Ax, Ay, Bx,
		time,
		mdl->total,
		istart, ystart, wstart,
		use->var_d, use->weight_d);
};

// =======================================================
// ================= Forward_t ===========================
// =======================================================

void lstm2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 locdstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	lstm2d_forward_th11<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16), train->sets),dim3(16,16,1)>>>(
		Ax, Ay, Bx,
		time,
		istart, ystart, wstart, locdstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd,
		inst * start_seed, drate/100,
		train->sets);
}

// =======================================================
// =================== Backward_t ========================
// =======================================================

void lstm2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 locdstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	lstm2d_backward_th11<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16), train->sets),dim3(16,16,1)>>>(	//backward input (size = Ax*Ay)
		Ax, Ay, Bx,
		time,
		istart, ystart, wstart, locdstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, ((float)drate)/100,
		train->sets);

	/*
	//	Backward input and .W     with kernel projected on input coords(Ax,Ay)
	lstm2d_backward_INPUT_th1x1<<<dim3(KERN_DIV(Ax,16),KERN_DIV(Ay,16), train->sets),dim3(16,16,1)>>>(	//backward input (size = Ax*Ay)
		Ax, Ay, Bx,
		time,
		istart, ystart, wstart, locdstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, ((float)drate)/100,
		train->sets);

	//	Backward h[-1] .U and .B     with kernel projected on `h` coords(Bx,Ay)
	if (time == 0) {	//because at time==0 => h[-1] = 0
		lstm2d_backward_BIAS_ONLY_th1x1<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16), train->sets),dim3(16,16,1)>>>(	//backward output (size = Bx * Ay)
			Ax, Ay, Bx,
			time,
			istart, ystart, wstart, locdstart,
			total, wsize, locdsize,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, ((float)drate)/100,
			train->sets);
	} else {
		lstm2d_backward_H1_BIAS_th1x1<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16), train->sets),dim3(16,16,1)>>>(	//backward output (size = Bx * Ay)
			Ax, Ay, Bx,
			time,
			istart, ystart, wstart, locdstart,
			total, wsize, locdsize,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, ((float)drate)/100,
			train->sets);
	}*/
}