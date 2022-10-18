#include "kernel/head/etc.cuh"

void etc_parse_arguments(uint argc, char ** argv, uint paramc, const char ** paramv, char ** correspondance) {
	for (uint i=0; i < argc; i++) {
		//	Find the correspondance
		for (uint j=0; j < paramc; j++) {
			if (strcmp(argv[i+1], paramv[j]) == 0) {	//'-sets 3' to skip the '-'
				correspondance[j] = argv[i];
				i++;
				break;
			}
		}

	}
};
