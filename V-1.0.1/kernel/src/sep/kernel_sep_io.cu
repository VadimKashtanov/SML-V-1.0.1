#include "kernel/head/sep.cuh"

Separators_t * sep_fp_load(FILE * fp)
{
	Separators_t * ret = (Separators_t*)malloc(sizeof(Separators_t));

	fread(&ret->seps, sizeof(uint), 1, fp);

	ret->sep_pos = (uint*)malloc(sizeof(uint) * ret->seps);
	ret->labels = (char**)malloc(sizeof(char*) * ret->seps);

	uint len;
	for (uint s=0; s < ret->seps; s++) {
		//	Load Label
		fread(&len, sizeof(uint), 1, fp);
		ret->labels[s] = (char*)malloc(sizeof(char) * (len+1));
		fread(ret->labels[s], sizeof(char), len, fp);
		ret->labels[s][len] = '\0';

		//	Load position
		fread(&ret->sep_pos[s], sizeof(uint), 1, fp);
	};

	return ret;
};

void sep_fp_write(FILE * fp, Separators_t * sep) {
	fwrite(&sep->seps, sizeof(uint), 1, fp);

	uint len;
	for (uint s=0; s < sep->seps; s++) {
		//	Load Label
		len = strlen(sep->labels[s]);
		fwrite(&len, sizeof(uint), 1, fp);
		fwrite(sep->labels[s], sizeof(char), len, fp);
		
		//	Load position
		fwrite(&sep->sep_pos[s], sizeof(uint), 1, fp);
	};
};
