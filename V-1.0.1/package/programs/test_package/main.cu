#include "package/package.cuh"

#define PRINT_ALL true

#define PYTHON_DEBBUG false

int main(int argc, char ** argv) {
	system("rm save.bin");

	printf("###################################################################\n");
	printf("################ Python Is Writing in `save.bin` ##################\n");
	printf("###################################################################\n");

#if PYTHON_DEBBUG == false
	system("python3 -m package.programs.test_package.run");	//will write into save.bin
#else
	system("python3 -m pdb -m package.programs.test_package.run");
#endif

	printf("###################################################################\n");
	printf("################ Testing All Data with C/Cuda #####################\n");
	printf("###################################################################\n");

	FILE * fp = fopen("save.bin", "rb");

	if (fp == 0)
		ERR("save.bin doesn't exists")
	
	printf("\033[43m");
	printf("=======================================================\n");
	printf("=================== Test Test Models ==================\n");
	printf("=======================================================\n");
	printf("\033[0m");
	uint test_mdls;
	fread(&test_mdls, sizeof(uint), 1, fp);
	for (uint m=0; m < test_mdls; m++) {
		printf("===============================================\n");
		printf("=================== Model \033[96m %i \033[0m ==================\n", m);
		printf("===============================================\n");
		test_mdl(PRINT_ALL ,fp);
	}

	printf("\033[46m");
	printf("=======================================================\n");
	printf("=================== Test Scores =======================\n");
	printf("=======================================================\n");
	printf("\033[0m");
	uint test_scores;
	fread(&test_scores, sizeof(uint), 1, fp);
	for (uint s=0; s < test_scores; s++) {
		printf("===============================================\n");
		printf("=================== Score \033[96m %i \033[0m ==================\n", s);
		printf("===============================================\n");
		test_score(PRINT_ALL ,fp);
	}

	printf("\033[45m");
	printf("=======================================================\n");
	printf("=================== Test Optimizers ===================\n");
	printf("=======================================================\n");
	printf("\033[0m");
	uint test_optis;
	fread(&test_optis, sizeof(uint), 1, fp);
	for (uint o=0; o < test_optis; o++) {
		printf("===============================================\n");
		printf("=================== Opti \033[96m %i \033[0m ===================\n", o);
		printf("===============================================\n");
		test_opti(PRINT_ALL, fp);
	}

	fclose(fp);
	printf("****************************\n");
	printf("All tests passed succesfully\n");
	printf("****************************\n");
}