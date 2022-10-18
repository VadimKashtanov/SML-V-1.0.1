#include "kernel/head/train.cuh"

static void vsep_compare_cpu_cpu(Separators_t * sep, float * arr0, float * arr1, uint sets, uint lines, uint total)
{
	int lbl;
	uint pos;

	for (uint l=0; l < lines; l++) {
		printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));
		printf("Line = %i ################### \n", l);
		for (uint s=0; s < sets; s++) {
			printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
			printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
			printf("Set #%i ============= \n", s);
			for (uint i=0; i < total; i++) {
				lbl = find_sep(sep, i);

				if (lbl != -1) {
					printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
					printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
					printf("|| (%i) %s\n", i, sep->labels[lbl]);
				}

				pos = l*total*sets + s*total + i;
				
				printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
				
				if (compare_floats(arr0[pos], arr1[pos], COMPARE_DEEPH)) 
					printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
				else
					printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
			}
		}
	}
};

static void wsep_compare_cpu_cpu(Separators_t * sep, float * arr0, float * arr1, uint sets, uint weights)
{
	int lbl;
	uint pos;

	for (uint s=0; s < sets; s++) {
		printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
		printf("Set #%i ============= \n", s);
		for (uint i=0; i < weights; i++) {
			lbl = find_sep(sep, i);

			if (lbl != -1) {
				printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
				printf("|| (%i) %s\n", i, sep->labels[lbl]);
			}

			pos = weights*s + i;
				
			printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
				
			if (compare_floats(arr0[pos], arr1[pos], COMPARE_DEEPH)) 
				printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
			else
				printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
		}
	}
};

static void lsep_compare_cpu_cpu(Separators_t * sep, float * arr0, float * arr1, uint sets, uint lines, uint locds)
{
	int lbl;
	uint pos;

	for (uint l=0; l < lines; l++) {
		printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));
		printf("Line = %i ################### \n", l);
		for (uint s=0; s < sets; s++) {
			printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
			printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
			printf("Set #%i ============= \n", s);
			for (uint i=0; i < locds; i++) {
				lbl = find_sep(sep, i);

				if (lbl != -1) {
					printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
					printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
					printf("|| (%i) %s\n", i, sep->labels[lbl]);
				}

				pos = l*locds*sets + s*locds + i;
				
				printf("\033[%im||\033[0m", (l % 2 ? 92 : 91));	// '||' de la ligne
				printf("\033[%im||\033[0m", (s % 2 ? 93 : 96)); // '||' du set
				
				if (compare_floats(arr0[pos], arr1[pos], COMPARE_DEEPH)) 
					printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
				else
					printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, arr0[pos], arr1[pos]);
			}
		}
	}
};

//=======================================================================================

void train_compare_weights(Train_t * train, float * with_this) {
	uint sets = train->sets;
	uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	//uint lines = train->data->lines;

	float * tmpt = (float*)malloc(sizeof(float) * (sets * weights));
	SAFE_CUDA(cudaMemcpy(tmpt, train->_weight, sizeof(float) * (sets * weights), cudaMemcpyDeviceToHost));
	wsep_compare_cpu_cpu(train->mdl->wsep, tmpt, with_this, sets, weights);
	free(tmpt);

	printf("             C/Cuda  ||| Python\n");
};

void train_compare_vars(Train_t * train, float * with_this) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * tmpt = (float*)malloc(sizeof(float) * (sets * lines * total));
	SAFE_CUDA(cudaMemcpy(tmpt, train->_var, sizeof(float) * (sets * lines * total), cudaMemcpyDeviceToHost));
	vsep_compare_cpu_cpu(train->mdl->vsep, tmpt, with_this, sets, lines, total);
	free(tmpt);

	printf("             C/Cuda  ||| Python\n");
};

void train_compare_locds(Train_t * train, float * with_this) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * tmpt = (float*)malloc(sizeof(float) * (sets * lines * locds));
	SAFE_CUDA(cudaMemcpy(tmpt, train->_locd, sizeof(float) * (sets * lines * locds), cudaMemcpyDeviceToHost));
	lsep_compare_cpu_cpu(train->mdl->lsep, tmpt, with_this, sets, lines, locds);
	free(tmpt);

	printf("             C/Cuda  ||| Python\n");
};

void train_compare_grads(Train_t * train, float * with_this) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * tmpt = (float*)malloc(sizeof(float) * (sets * lines * total));
	SAFE_CUDA(cudaMemcpy(tmpt, train->_grad, sizeof(float) * (sets * lines * total), cudaMemcpyDeviceToHost));
	vsep_compare_cpu_cpu(train->mdl->vsep, tmpt, with_this, sets, lines, total);
	free(tmpt);

	printf("             C/Cuda  ||| Python\n");
};

void train_compare_meands(Train_t * train, float * with_this) {
	uint sets = train->sets;
	uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	//uint lines = train->data->lines;

	float * tmpt = (float*)malloc(sizeof(float) * (sets * weights));
	SAFE_CUDA(cudaMemcpy(tmpt, train->_meand, sizeof(float) * (sets * weights), cudaMemcpyDeviceToHost));
	wsep_compare_cpu_cpu(train->mdl->wsep, tmpt, with_this, sets, weights);
	free(tmpt);

	printf("           C/Cuda  ||| Python\n");
};
