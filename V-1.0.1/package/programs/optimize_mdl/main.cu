#include "package/package.cuh"

/*
        0          1       2       3       4     5      6    7       8
./optimize_mdl data.bin mdl.bin out.bin echopes repeat sets score optimizer
	
	Score : meansquared, crossentropy
	Optimizer : sgd, moment, rmsprop, adam
*/

#define PERCENT_TEST_BATCHS 1.0

char * data_file = 0;
char * mdl_file = 0;
char * out_file = 0;
uint echopes = 1;
uint repeat = 1;
uint sets = 1;
uint score_algo = 0;
uint opti_algo = 0;
uint test_batchs = 1;
uint __select = 0;
uint test_all_batchs = 0;
uint echo = 0;

uint eq_pos(char * arg) {
	uint i=0;
	char c = arg[0];
	while (1) {
		if (c == '=')
			return i;
		if (c == '\0')
			ERR("No '=' in {%s}", arg);
		i++;
		c = arg[i];
	}
	return 0;
};

void analyse_args(int argc, char ** argv) {
	uint __ep_pos;

	uint _len;
	for (int i=1; i < argc; i++) {
		__ep_pos = eq_pos(argv[i]);
		argv[i][__ep_pos] = '\0';

		if (strcmp(argv[i], "data_file") == 0) {
			_len = strlen(argv[i] + __ep_pos+1);
			data_file = (char*)malloc(sizeof(char) * (_len + 1));
			strcpy(data_file, argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "mdl_file") == 0) {
			_len = strlen(argv[i] + __ep_pos+1);
			mdl_file = (char*)malloc(sizeof(char) * (_len + 1));
			strcpy(mdl_file, argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "out_file") == 0) {
			_len = strlen(argv[i] + __ep_pos+1);
			out_file = (char*)malloc(sizeof(char) * (_len + 1));
			strcpy(out_file, argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "echopes") == 0) {
			echopes = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "repeat") == 0) {
			repeat = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "sets") == 0) {
			sets = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "score_algo") == 0) {
			score_algo = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "opti_algo") == 0) {
			opti_algo = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "test_batchs") == 0) {
			test_batchs = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "select") == 0) {
			__select = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "test_all_batchs") == 0) {
			test_all_batchs = atoi(argv[i]+__ep_pos+1);
		} else if (strcmp(argv[i], "echo") == 0) {
			echo = atoi(argv[i]+__ep_pos+1);
		} else {
			ERR("What is %s ?", argv[i]);
		}
	}
};

uint find_min(float * arr, uint len) {
	uint __min_id = 0;

	for (uint i=1; i < len; i++)
		if (arr[__min_id] > arr[i])
			__min_id = i;

	return __min_id;
}

#define ECHOPES_PRINT 20

int main(int argc, char ** argv) {

	analyse_args(argc, argv);

	FILE * mdlfp = fopen(mdl_file, "rb");
	Mdl_t * mdl = mdl_fp_load(mdlfp);
	fclose(mdlfp);

	//// Load to Ram and Vram
	Data_t * data = data_open(data_file);
	data_cudmalloc(data);

	uint test_batchs = floor(PERCENT_TEST_BATCHS * data->batchs + 0.5);

	FILE * data_fp = fopen(data_file, "rb");

	//// Build Train_t and Opti_t
	if (sets == 0)
		ERR("sets can't be = to 0")
	Train_t * train = mk_train(mdl, data, sets);
	train_random_weights(train, rand()%10000);

	Opti_t * opti = opti_mk(train, score_algo, opti_algo);

	float set_score_tests[sets];

	uint start_seed;

	uint batch_train;
	
	//	On veut print l'échope que le program train, mais en Max ECHOPES_PRINT fois.
	//	Donc on le print tout les echopes/ECHOPES_PRINT fois. Sauf Si echopes < ECHOPES_PRINT.
	//	Dans ce cas on print echopes fois.
	uint _tmpt = ceil((float)echopes / ECHOPES_PRINT);	//comme round mais donne toujours au dessus

	for (uint lp=0; lp < echopes; lp++) {
		//Loop
		batch_train = rand() % data->batchs;

		//	Load a batch
		data_load_batch(data, data_fp, batch_train);
		data_cudamemcpy(data);

		train_set_input(train);

		//	Trainning Part
		for (uint i=0; i < repeat; i++) {
			//	Initialise correctly
			train_set_input(train);
			train_null_grad_meand(train);
	
			//	Forward and Backward
			start_seed = rand() % 100000;

			train_forward(train, start_seed);
			opti_dloss(opti);
			train_backward(train, start_seed);

			if (echo)
				train_print_meands(train);

			//	Optimize
			opti_opti(opti);
		}

		if (echopes < ECHOPES_PRINT || ((lp+1) % _tmpt) == 0) {
			printf("Echope : %i/%i [batch=%i]\n", lp, echopes, batch_train);

			/*for (uint s=0; s < sets; s++)
				set_score_tests[s] = 0;

			for (uint i=0; i < test_batchs; i++) {
				//	Select the best
				batch_train = rand() % data->batchs;
					
				data_load_batch(data, data_fp, batch_train);
				data_cudamemcpy(data);
				opti_loss(opti);
				
				for (uint s=0; s < sets; s++)
					set_score_tests[s] += opti->set_score[s] / test_batchs;
			}

			for (uint s=0; s < sets; s++) {
				printf("%i|\033[93m %f \033[0m\n", s, set_score_tests[s]);
			}*/

		}
	};

	printf("Echope : %i/%i [batch=%i]\n", echopes, echopes, batch_train);

	////	Compute Score
	for (uint s=0; s < sets; s++)
		set_score_tests[s] = 0;

	if (test_all_batchs) {
		for (uint i=0; i < data->batchs; i++) {
			data_load_batch(data, data_fp, i);
			data_cudamemcpy(data);
			opti_loss(opti);
			
			for (uint s=0; s < sets; s++)
				set_score_tests[s] += opti->set_score[s] / data->batchs;
		}
	} else {
		for (uint i=0; i < test_batchs; i++) {
			//	Select the best
			batch_train = rand() % data->batchs;
				
			data_load_batch(data, data_fp, batch_train);
			data_cudamemcpy(data);
			opti_loss(opti);
			
			for (uint s=0; s < sets; s++)
				set_score_tests[s] += opti->set_score[s] / test_batchs;
		}
	}

	printf("## Scores ##\n");
	for (uint s=0; s < sets; s++) {
		//set_score_tests[s] /= test_batchs;
		printf("|| %i | \033[93m %f \033[0m\n", s, set_score_tests[s]);
	}

	//	Take Best set
	uint best_set = find_min(set_score_tests, sets);//opti->podium[0];

	train_cpy_ws_to_mdl(train, best_set);

	mdlfp = fopen(out_file, "wb");
	mdl_fp_write(mdl, mdlfp);
	fclose(mdlfp);

	//	Free all to make a correct valgrind and juste to make all clean (and each malloc have to be freed)
	opti_free(opti);
	train_free(train);
	data_free(data);
	mdl_free(mdl);
};