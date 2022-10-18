#include "package/scores/crossentropy/head/crossentropy.cuh"

//=================================================================================================
//===================================== dLOSS(g,w)/dg =============================================
//=================================================================================================

static __global__ void opti_kernel_ce_dloss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs,
	uint sets)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;

		float __out = var[pos];

		if (__out == 0) {
			printf("\033[91mThere is 0 values with crossentropy. Division by 0 doesn't exists. Make sure last instruction produce >0 values only.\033[0m\n");
			assert(0);
		}

		grad[pos] = -output[line*outs + out]/__out;
	};
};

void CROSSENTROPY_dloss(Opti_t * opti) {
	Train_t * train = opti->train;

	uint outpos = train->mdl->total - train->mdl->outputs;

	opti_kernel_ce_dloss<<<dim3(KERN_DIV(train->mdl->outputs, 32), KERN_DIV(train->data->lines, 32), train->sets),dim3(32, 32, 1)>>>(
		train->_grad, train->_var, train->data->output_d,
		train->mdl->total, outpos, train->data->lines, train->data->outputs,
		train->sets
	);
	SAFE_CUDA(cudaPeekAtLastError());
};

//=================================================================================================
//====================================== LOSS(g,w) ================================================
//=================================================================================================

static __global__ void opti_kernel_ce_loss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs,
	uint sets)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		float g = var[pos];
		float w = output[line*outs + out];

		if (g == 0) {
			//on peut pas utiliser ERR() dans les fonctions __host__
			printf("\033[91mThere is 0 values with crossentropy. log(0) doesn't exists. Make sure last instruction produce >0 values only.\033[0m\n");
			assert(0);
		}

		grad[pos] = -w*log(g);
	};
};

static __global__ void opti_kernel_sum_scores_over_lines(
	float * grad, float * var, float * output,
	float * score_one_line_d,
	uint total, uint lines, uint sets, uint ostart, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (out < outs)
	{
		//uint pos;
		float _sum_of_lines = 0;
		for (uint l=0; l < lines; l++) {
			_sum_of_lines += grad[l*sets*total + set*total + ostart + out];
		}
		score_one_line_d[set*total + out] = _sum_of_lines / lines;
	};
};

static __global__ void opti_kernel_sum_scores_over_outputs(
	float * score_one_line_d, float * scores,
	uint total, uint sets, uint ostart, uint outs)
{
	uint set = blockIdx.x;

	uint start = set*total + 0;
	float _sum_of_outs = 0;
	for (uint o=0; o < outs; o++) {
		_sum_of_outs += score_one_line_d[start];
		start++;
	}

	scores[set] = _sum_of_outs / outs;
};

void CROSSENTROPY_loss(Opti_t * opti) {
	Train_t * train = opti->train;
	Mdl_t * mdl = train->mdl;

	uint outs = mdl->outputs;
	uint lines = train->data->lines;
	uint sets = train->sets;
	uint out_start = mdl->total - outs;

	//======================================================================

	//						compute score

	opti_kernel_ce_loss<<<dim3(KERN_DIV(outs, 32), KERN_DIV(lines, 32), sets),dim3(32,32,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, out_start, lines, outs,
		sets);
	SAFE_CUDA(cudaPeekAtLastError());

	//======================================================================
	//======================================================================

	//				sum over lines (only outputs)

	float * score_one_line_d;
	SAFE_CUDA(cudaMalloc((void**)&score_one_line_d, sizeof(float) * sets * outs));	//all lines are sumed in one (only outputs)

	opti_kernel_sum_scores_over_lines<<<dim3(KERN_DIV(outs, 16), sets),dim3(16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		score_one_line_d,
		mdl->total, lines, sets, out_start, outs);
	SAFE_CUDA(cudaPeekAtLastError());

	//======================================================================
	//======================================================================

	//		sum of output pixels

	opti_kernel_sum_scores_over_outputs<<<dim3(sets),dim3(1)>>>(
		score_one_line_d, opti->set_score_d,
		mdl->total, sets, out_start, outs);
	SAFE_CUDA(cudaPeekAtLastError());

	SAFE_CUDA(cudaFree(score_one_line_d));
};