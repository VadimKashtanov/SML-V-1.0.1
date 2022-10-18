#include "kernel/head/testpackage.cuh"

void test_opti(bool print_all, FILE * fp) {
	//	Load Contexte
	Mdl_t * mdl = mdl_fp_load(fp);
	read_123(fp);

	Data_t * data = load_test_data(fp);
	read_123(fp);

	//uint total = mdl->total;
	//uint lines = data->lines;
	//uint locds = mdl->locds;
	//uint weights = mdl->weights;

	uint sets;
	fread(&sets, sizeof(uint), 1, fp);
	read_123(fp);

	Train_t * train = mk_train(mdl, data, sets);
	train_random_weights(train, 0);

	uint score_algo, opti_algo;
	fread(&score_algo, sizeof(uint), 1, fp);
	fread(&opti_algo, sizeof(uint), 1, fp);
	read_123(fp);

	Opti_t * opti_space = opti_mk(train, score_algo, opti_algo);	//0 == any optimizer

	opti_score_load_consts(opti_space, fp);
	read_123(fp);

	opti_opti_load_consts(opti_space, fp);
	read_123(fp);

	//loss and dloss
	uint loops;	//optimizers have to make not only one loop to fully use the algorithm.
	fread(&loops, sizeof(uint), 1, fp);
	read_123(fp);

	for (uint l=0; l < loops; l++) {
		train_null_grad_meand(train);
		train_set_input(train);
		train_forward(train, 0);
		opti_dloss(opti_space);
		train_backward(train, 0);
		opti_opti(opti_space);
	}
	///

	float * compare_array;

	//	Meands
	compare_array = load_float_array(sets * mdl->weights, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_meand, sets*mdl->weights)) {
		if (print_all)
			train_compare_meands(train, compare_array);
		OK("Train_t->_meand passed correctly.");
	} else {
		train_compare_meands(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_meand apres l'optimizeur, au bout de %i boucles", loops);
	}
	free(compare_array);
	read_123(fp);

	//	Weights
	compare_array = load_float_array(sets * mdl->weights, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_weight, sets*mdl->weights)) {
		if (print_all)
			train_compare_weights(train, compare_array);
		OK("Train_t->_weight passed correctly.");
	} else {
		train_compare_meands(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_weight apres l'optimizeur, au bout de %i boucles", loops);
	}
	free(compare_array);
	read_123(fp);


	//free
	//gtic_free(gtic);
	opti_free(opti_space);
	train_free(train);
	data_free(data);
	mdl_free(mdl);
};