#include "kernel/head/testpackage.cuh"

void test_score(bool print_all, FILE * fp) {
	//	Load Contexte
	Mdl_t * mdl = mdl_fp_load(fp);
	read_123(fp);

	Data_t * data = load_test_data(fp);
	read_123(fp);

	uint total = mdl->total;
	uint lines = data->lines;
	//uint locds = mdl->locds;
	//uint weights = mdl->weights;

	uint sets;
	fread(&sets, sizeof(uint), 1, fp);
	read_123(fp);

	Train_t * train = mk_train(mdl, data, sets);
	train_random_weights(train, 0);

	uint score_algo;
	fread(&score_algo, sizeof(uint), 1, fp);
	read_123(fp);

	Opti_t * opti_space = opti_mk(train, score_algo, 0);	//0 == any optimizer

	opti_score_load_consts(opti_space, fp);
	read_123(fp);

	//loss and dloss
	train_null_grad_meand(train);
	train_set_input(train);
	train_forward(train, 0);

	float * compare_array;

	//loss()
	{
		opti_dloss(opti_space);
		train_backward(train, 0);
		//		train->_var
		compare_array = load_float_array(lines * sets * total, fp);
		if (test_package_compare_cpu_and_gpu(compare_array, train->_grad, lines*sets*total)) {
			if (print_all)
				train_compare_grads(train, compare_array);
			OK("Train_t->_grad passed correctly.");
		} else {
			train_compare_grads(train, compare_array);
			ERR("Il y a des Erreures avec Train_t->_grad (apres dloss)")
		}
		free(compare_array);
		read_123(fp);
	}

	//dloss()
	{
		opti_loss(opti_space);

		//		train->_grad
		compare_array = load_float_array(lines * sets * total, fp);
		if (test_package_compare_cpu_and_gpu(compare_array, train->_grad, lines*sets*total)) {
			if (print_all)
				train_compare_grads(train, compare_array);
			OK("Train_t->_grad passed correctly.");
		} else {
			train_compare_grads(train, compare_array);
			ERR("Il y a des Erreures avec Train_t->_grad (apres loss)")
		}
		free(compare_array);
		read_123(fp);

		//		opti_space->set_score
		compare_array = load_float_array(sets, fp);
		if (test_package_compare_cpu_and_cpu(compare_array, opti_space->set_score, sets)) {
			if (print_all)
				opti_compare_scores(opti_space, compare_array);
			OK("Opti_t->set_score passed correctly.");
		} else {
			opti_compare_scores(opti_space, compare_array);
			ERR("Il y a des Erreures avec Opti_t->set_score")
		}
		free(compare_array);
		read_123(fp);
	}

	//free
	//gtic_free(gtic);
	opti_free(opti_space);
	train_free(train);
	data_free(data);
	mdl_free(mdl);
};