#include "package/insts/softmax/head/softmax.cuh"

/*
for i in range(_len):
	err = errors[i]
			
	for j in range(_len):
		yi = var[i]
		yj = var[j]

		if i == j:
			grad[j] += err * yi * (1 - yi)
		else:
			grad[j] += - err * yi * yj

Sauf que j'ai inverser i et j, pour ne pas faire de atomicAdd(&) beaucoup trop

for j in range(_len):
	_grad = 0
	for i in range(_len):
		yi = var[i]
		yj = var[j]
			
		if i == j:
			_grad += errors[i] * yi * (1 - yi)
		else:
			_grad += - errors[i] * yi * yj
	grad[j] = _grad
*/

__global__
void softmax_backward_th32(
	uint len,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart,
	uint sets,
	float * var, float * grad)
{
	uint j = threadIdx.x;
	uint set = blockIdx.x;

	if (j < len) {
		float _grad = 0;

		__shared__ float _y[32];
		__shared__ float _err[32];

		_y[j] = var[time*sets*total + set*total + ystart + j];
		_err[j] = grad[time*sets*total + set*total + ystart + j];

		__syncthreads();

		//	Au lieu de faire une boucle avec un if a chaque fois, on fait 2 boucle et le seul cas quand i==j
		//Mais de toute façon ça reste
		//	(i==j) err[j] * yj * (1 - yj)
		//	(i!=i) err[i] * yi * yj

		for (uint i=0; i < j; i++)
			_grad += -_err[i] * _y[i] * _y[j];

		//err[j] et pas err[i] car i == j (sauf que `i` est utilisé que dans la boucle for)
		_grad += _err[j] * _y[j] * (1-_y[j]);	//car i==j

		for (uint i=j+1; i < len; i++)
			_grad += -_err[i] * _y[i] * _y[j];

		//on met dans le grad
		grad[time*sets*total + set*total + istart + j] = _grad;	//input_start car la on fait le gradient de l'input (car l'erreur (grad output) est déjà calculé)
	}
}