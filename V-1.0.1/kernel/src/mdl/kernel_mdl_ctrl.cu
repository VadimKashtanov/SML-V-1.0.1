#include "kernel/head/mdl.cuh"

void mdl_check_correctness(Mdl_t * mdl) {
	//	Pour l'instant on met une limite d'instructions de 1000
	if (mdl->insts > 1000)
		raise(SIGINT);

	//	On check si les informations sur les instructions sont coherantes
	for (uint i=0; i < mdl->insts; i++) {
		if (mdl->id[i] >= INSTS)
			raise(SIGINT);

		INST_CHECK[mdl->id[i]](mdl->param[i]);
	}

	//	On verifie que mdl->total est bien calculé
	if (mdl->total != mdl->inputs + mdl->vars)
		raise(SIGINT);

	//	On verifie que les Separateur sont pas incoherants
	//vsep
	for (uint i=0; i < mdl->vsep->seps; i++)
		if (mdl->vsep->sep_pos[i] >= mdl->total)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
	//wsep
	for (uint i=0; i < mdl->wsep->seps; i++)
		if (mdl->wsep->sep_pos[i] >= mdl->weights)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
	//lsep
	for (uint i=0; i < mdl->lsep->seps; i++)
		if (mdl->lsep->sep_pos[i] >= mdl->locds)
			raise(SIGINT);	//la position ne peut pas etre apres la ligne.
};

