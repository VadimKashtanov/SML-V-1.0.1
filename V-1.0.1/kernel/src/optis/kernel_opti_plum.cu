#include "kernel/head/optis.cuh"

void opti_print_scores(Opti_t * opti) {
	for (uint i=0; i < opti->train->sets; i++)
		printf("|| %i |  \033[93m %f \033[0m \n", i, opti->set_score[i]);
};

void opti_compare_scores(Opti_t * opti, float * with_this) {
	printf("	mdl->weight   compare_array\n");
	for (uint i=0; i < opti->train->sets; i++) {
		if (compare_floats(opti->set_score[i], with_this[i], 0.001)) 
			printf("|| %i |  \033[42m %f --- %f \033[0m \n", i, opti->set_score[i], with_this[i]);
		else
			printf("|| %i |  \033[41m %f --- %f \033[0m \n", i, opti->set_score[i], with_this[i]);
	}

	printf("             C/Cuda  ||| Python\n");
};

void opti_print_rank(Opti_t * opti) {
	for (uint i=0; i < opti->train->sets; i++)
		printf("|| %i |  \033[93m %i'th rank \033[0m \n", i, opti->set_rank[i]);
};

void opti_print_podium(Opti_t * opti) {
	for (uint i=0; i < opti->train->sets; i++)
		printf("|| %i |  \033[93m %i'th set \033[0m \n", i, opti->podium[i]);
};
