#include "kernel/head/train.cuh"

static void vsep_print(Separators_t * sep, float * arr0, uint sets, uint lines, uint total)
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
				
				printf("|| %i |  \033[93m %f \033[0m \n", i, arr0[pos]);
			}
		}
	}
};

static void wsep_print(Separators_t * sep, float * arr0, uint sets, uint weights)
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
				
			printf("|| %i |  \033[93m %f \033[0m \n", i, arr0[pos]);
		}
	}
};

static void lsep_print(Separators_t * sep, float * arr0, uint sets, uint lines, uint locds)
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
				
				printf("|| %i |  \033[93m %f \033[0m \n", i, arr0[pos]);
			}
		}
	}
};

//	---------------------------------------------------------------------------------------------

void train_print_weights(Train_t * train) {
	uint sets = train->sets;
	uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	//uint lines = train->data->lines;

	float * arr1 = (float*)malloc(sizeof(float) * (sets * weights));
	SAFE_CUDA(cudaMemcpy(arr1, train->_weight, sizeof(float) * (sets * weights), cudaMemcpyDeviceToHost));
	wsep_print(train->mdl->wsep, arr1, sets, weights);
	free(arr1);
};

void train_print_vars(Train_t * train) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * arr1 = (float*)malloc(sizeof(float) * (sets * lines * total));
	SAFE_CUDA(cudaMemcpy(arr1, train->_var, sizeof(float) * (sets * lines * total), cudaMemcpyDeviceToHost));
	vsep_print(train->mdl->vsep, arr1, sets, lines, total);
	free(arr1);
};

void train_print_locds(Train_t * train) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * arr1 = (float*)malloc(sizeof(float) * (sets * lines * locds));
	SAFE_CUDA(cudaMemcpy(arr1, train->_locd, sizeof(float) * (sets * lines * locds), cudaMemcpyDeviceToHost));
	lsep_print(train->mdl->lsep, arr1, sets, lines, locds);
	free(arr1);
};

void train_print_grads(Train_t * train) {
	uint sets = train->sets;
	//uint weights = train->mdl->weights;
	uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	uint lines = train->data->lines;

	float * arr1 = (float*)malloc(sizeof(float) * (sets * lines * total));
	SAFE_CUDA(cudaMemcpy(arr1, train->_grad, sizeof(float) * (sets * lines * total), cudaMemcpyDeviceToHost));
	vsep_print(train->mdl->vsep, arr1, sets, lines, total);
	free(arr1);
};

void train_print_meands(Train_t * train) {
	uint sets = train->sets;
	uint weights = train->mdl->weights;
	//uint total = train->mdl->total;
	//uint locds = train->mdl->locds;
	//uint lines = train->data->lines;

	float * arr1 = (float*)malloc(sizeof(float) * (sets * weights));
	SAFE_CUDA(cudaMemcpy(arr1, train->_meand, sizeof(float) * (sets * weights), cudaMemcpyDeviceToHost));
	wsep_print(train->mdl->wsep, arr1, sets, weights);
	free(arr1);
};

//
void train_print_all(Train_t * train) {
	printf(" ============== WEIGHTS ==============\n");
	train_print_weights(train);
	printf(" ============== VARS ==============\n");
	train_print_vars(train);
	printf(" ============== LOCDS ==============\n");
	train_print_locds(train);
	printf(" ============== GRADS ==============\n");
	train_print_grads(train);
	printf(" ============== MEANDS ==============\n");
	train_print_meands(train);
};
