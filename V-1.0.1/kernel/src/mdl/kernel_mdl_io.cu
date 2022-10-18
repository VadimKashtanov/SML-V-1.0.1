#include "kernel/head/mdl.cuh"

Mdl_t* mdl_fp_load(FILE * fp) {
	Mdl_t * ret = (Mdl_t*)malloc(sizeof(Mdl_t));

	/*			Instructions		*/
	fread(&ret->insts, sizeof(uint), 1, fp);

	ret->id = (uint*)malloc(sizeof(uint) * ret->insts);
	ret->param = (uint**)malloc(sizeof(uint*) * ret->insts);

	for (uint i=0; i < ret->insts; i++) {
		//Instruction Id
		fread(&ret->id[i], sizeof(uint), 1, fp);

		//Parameters
		ret->param[i] = (uint*)malloc(sizeof(uint) * inst_params[ret->id[i]]);
		fread(ret->param[i], sizeof(uint), inst_params[ret->id[i]], fp);
	}

	fread(&ret->inputs, sizeof(uint), 1, fp);
	fread(&ret->outputs, sizeof(uint), 1, fp);

	fread(&ret->vars, sizeof(uint), 1, fp);
	fread(&ret->weights, sizeof(uint), 1, fp);
	fread(&ret->locds, sizeof(uint), 1, fp);

	ret->weight = (float*)malloc(sizeof(float) * ret->weights);
	
	fread(ret->weight, sizeof(float), ret->weights, fp);

	ret->total = ret->inputs + ret->vars;

	//	Separators
	ret->vsep = sep_fp_load(fp);
	ret->wsep = sep_fp_load(fp);
	ret->lsep = sep_fp_load(fp);
	
	return ret;
};

void mdl_fp_write(Mdl_t * mdl, FILE * fp) {
	fwrite(&mdl->insts, sizeof(uint), 1, fp);

	for (uint i=0; i < mdl->insts; i++) {
		//Instruction Id
		fwrite(&mdl->id[i], sizeof(uint), 1, fp);
		fwrite(mdl->param[i], sizeof(uint), inst_params[mdl->id[i]], fp);
	}

	fwrite(&mdl->inputs, sizeof(uint), 1, fp);
	fwrite(&mdl->outputs, sizeof(uint), 1, fp);

	fwrite(&mdl->vars, sizeof(uint), 1, fp);
	fwrite(&mdl->weights, sizeof(uint), 1, fp);
	fwrite(&mdl->locds, sizeof(uint), 1, fp);

	fwrite(mdl->weight, sizeof(float), mdl->weights, fp);

	sep_fp_write(fp, mdl->vsep);
	sep_fp_write(fp, mdl->wsep);
	sep_fp_write(fp, mdl->lsep);
};
