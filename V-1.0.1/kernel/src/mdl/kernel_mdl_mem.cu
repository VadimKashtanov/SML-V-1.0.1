#include "kernel/head/mdl.cuh"

void mdl_free(Mdl_t * mdl) {
	//	Separators
	sep_free(mdl->vsep);
	sep_free(mdl->wsep);
	sep_free(mdl->lsep);
	
	//	Insts
	free(mdl->id);
	for (uint i=0; i < mdl->insts; i++)
		free(mdl->param[i]);
	free(mdl->param);

	//Ws
	free(mdl->weight);

	//
	free(mdl);
};
