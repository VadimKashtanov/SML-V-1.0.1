#include "kernel/head/optis.cuh"

void opti_opti(Opti_t * opti) {
	OPTI_OPTIMIZE_ARRAY[opti->opti_algo](opti);
};

void opti_loss(Opti_t * opti) {
	OPTI_COMPUTE_LOSS_ARRAY[opti->opti_algo](opti);
};

void opti_dloss(Opti_t * opti) {
	OPTI_SCORES_DLOSS_ARRAY[opti->opti_algo](opti);
};