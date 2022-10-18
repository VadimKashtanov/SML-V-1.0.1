#include "package/optis/sgd/head/sgd.cuh"

__global__
void sgd_kernel_th11(
	float sgd_alpha,
	uint weights, uint lines,
	float * weight, float * meand)
{
	uint w = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (w < weights)
		weight[set*weights + w] -= sgd_alpha * meand[set*weights + w] / lines;
};

void SGD_optimize(Opti_t * opti)
{
	SGDData_t * ret = (SGDData_t*)opti->opti_space;

	ret->echopes++;

	sgd_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_sgd_alpha,
		opti->train->mdl->weights, opti->train->data->lines,
		opti->train->_weight, opti->train->_meand
	);

	//opti_sgd_alpha *= ( 1 / ( 1 + 0.1/2000 * ret->echopes));

	cudaDeviceSynchronize();
	SAFE_CUDA(cudaPeekAtLastError());
};

//
//
//		Tester avec un probleme de beaucoup plus petite taille
//		Pas du Mnist, mais par exemple classifier 3 lettres A,B,C avec des images de taille 8x8
//		Pour kconvl 8x8->4x4->2x2->dot1d(4)->softmax(4)
//
//		1. Faire en python un petit programme qui ecrit les lettres en pixels avec l'IDLE
//		2. Faire un Data_t avec ces pixels
//		3. Cree le model simple et observer les trucs
//
//
//
//		Au lieux de fait un model qui predit plusieurs classes,
//		On train plusieur models d'une seule classe binaire.
//		Ex : On train le model qui predit un 0 ou un autre nombre que 0
//			puis On train le model qui predit un 1 ou un autre nombre que 1
//			puis le model qui predit un 2 ou un autre nombre que 2
//				..., jusqu'a 9
//		Apres on unie tout. On y ajoute un dot1->dot1d->softmax (en coupant le derniere dot1d->softmax des models precedants)
//		Bon ducoup ça train pas la diversité ensemble (des 0,1,2,3 .. n)
//		Mais ça va s'auto train juste apres tout seul.
//
//		(a la limite si les kernels bougent trop, on peut donner le resulats des model_precedant(input_qui_correspond) -> output)
//		(et le output sera mis dans le Data_t et puis plus tard, une fois train, on met ensemble tous les models)