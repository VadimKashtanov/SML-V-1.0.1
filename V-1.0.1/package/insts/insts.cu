#include "package/insts/insts.cuh"

uint inst_params[INSTS] = {
	8, //dot1d 
	9, //dot2d 
	10, //dot2drecurent 
	16, //kconvl2d 
	7, //pool2dmax 
	6, //pool2daverage 
	3, //softmax 
	6, //gaussfiltre2d 
	8, //lstm2d 
	4, //sum 
	6, //activation 
};

const char* inst_name[INSTS] = {
	"dot1d",
	"dot2d",
	"dot2drecurent",
	"kconvl2d",
	"pool2dmax",
	"pool2daverage",
	"softmax",
	"gaussfiltre2d",
	"lstm2d",
	"sum",
	"activation",
};

static const char* dot1d_params_names[8] = {
	"Ax",
	"Yx",
	"activ",
	"input_start",
	"ystart",
	"wstart",
	"locdstart",
	"drop_rate",
};

static const char* dot2d_params_names[9] = {
	"Ax",
	"Ay",
	"Bx",
	"activ",
	"input_start",
	"ystart",
	"wstart",
	"locdstart",
	"drop_rate",
};

static const char* dot2drecurent_params_names[10] = {
	"Ax",
	"Ay",
	"At",
	"Bx",
	"activ",
	"istart",
	"ystart",
	"wstart",
	"lstart",
	"drate",
};

static const char* kconvl2d_params_names[16] = {
	"Ax",
	"Ay",
	"Kx",
	"Ky",
	"n0",
	"n1",
	"strideX",
	"strideY",
	"paddingX",
	"paddingY",
	"activ",
	"input_start",
	"ystart",
	"wstart",
	"locdstart",
	"drop_rate",
};

static const char* pool2dmax_params_names[7] = {
	"Ax",
	"Ay",
	"Xpool",
	"Ypool",
	"input_start",
	"ystart",
	"locdstart",
};

static const char* pool2daverage_params_names[6] = {
	"Ax",
	"Ay",
	"Xpool",
	"Ypool",
	"input_start",
	"ystart",
};

static const char* softmax_params_names[3] = {
	"_len",
	"input_start",
	"ystart",
};

static const char* gaussfiltre2d_params_names[6] = {
	"X",
	"Y",
	"istart",
	"ystart",
	"wstart",
	"lstart",
};

static const char* lstm2d_params_names[8] = {
	"Ax",
	"Ay",
	"Bx",
	"istart",
	"ystart",
	"wstart",
	"locdstart",
	"drate",
};

static const char* sum_params_names[4] = {
	"size",
	"items",
	"istart",
	"ystart",
};

static const char* activation_params_names[6] = {
	"_len",
	"activ",
	"istart",
	"ystart",
	"wstart",
	"locdstart",
};

const char** inst_param_name[INSTS] = {
	dot1d_params_names,
	dot2d_params_names,
	dot2drecurent_params_names,
	kconvl2d_params_names,
	pool2dmax_params_names,
	pool2daverage_params_names,
	softmax_params_names,
	gaussfiltre2d_params_names,
	lstm2d_params_names,
	sum_params_names,
	activation_params_names,
};

check_f INST_CHECK[INSTS] = {
	dot1d_check,
	dot2d_check,
	dot2drecurent_check,
	kconvl2d_check,
	pool2dmax_check,
	pool2daverage_check,
	softmax_check,
	gaussfiltre2d_check,
	lstm2d_check,
	sum_check,
	activation_check,
};

cpu_f INST_CPU[INSTS] = {
	dot1d_cpu,
	dot2d_cpu,
	dot2drecurent_cpu,
	kconvl2d_cpu,
	pool2dmax_cpu,
	pool2daverage_cpu,
	softmax_cpu,
	gaussfiltre2d_cpu,
	lstm2d_cpu,
	sum_cpu,
	activation_cpu,
};

use_f INST_USE[INSTS] = {
	dot1d_use,
	dot2d_use,
	dot2drecurent_use,
	kconvl2d_use,
	pool2dmax_use,
	pool2daverage_use,
	softmax_use,
	gaussfiltre2d_use,
	lstm2d_use,
	sum_use,
	activation_use,
};

train_f INST_FORWARD[INSTS] = {
	dot1d_forward,
	dot2d_forward,
	dot2drecurent_forward,
	kconvl2d_forward,
	pool2dmax_forward,
	pool2daverage_forward,
	softmax_forward,
	gaussfiltre2d_forward,
	lstm2d_forward,
	sum_forward,
	activation_forward,
};

train_f INST_BACKWARD[INSTS] = {
	dot1d_backward,
	dot2d_backward,
	dot2drecurent_backward,
	kconvl2d_backward,
	pool2dmax_backward,
	pool2daverage_backward,
	softmax_backward,
	gaussfiltre2d_backward,
	lstm2d_backward,
	sum_backward,
	activation_backward,
};

