#include "kernel/head/optis.cuh"

void opti_score_load_consts(Opti_t * opti, FILE * fp) {
	uint consts;
	fread(&consts, sizeof(uint), 1, fp);

	uint _len;
	char * _tmp_char0;
	char * _tmp_char1;
	for (uint c=0; c < consts; c++) {
		{
			//	=== KEY ===
			fread(&_len, sizeof(uint), 1, fp);
			_tmp_char0 = (char*)malloc(_len + 1);
			fread(_tmp_char0, sizeof(char), _len, fp);
			_tmp_char0[_len] = '\0';

			//	=== VALUE ===
			fread(&_len, sizeof(uint), 1, fp);
			_tmp_char1 = (char*)malloc(_len + 1);
			fread(_tmp_char1, sizeof(char), _len, fp);
			_tmp_char1[_len] = '\0';
		}

		//	const[key] = value
		opti_score_set_one_arg(opti, _tmp_char0, _tmp_char1);

		//freeing
		free(_tmp_char0);
		free(_tmp_char1);
	};
};

void opti_score_set_one_arg(Opti_t * opti, char * argname, char * value) {
	OPTI_SCORE_SET_ONE_ARG_ARRAY[opti->score_algo](opti, argname, value);
};