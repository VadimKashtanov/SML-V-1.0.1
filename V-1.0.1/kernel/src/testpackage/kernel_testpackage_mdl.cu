#include "kernel/head/testpackage.cuh"

void test_mdl(bool print_all, FILE * fp) {
	//	Load Contexte
	Mdl_t * mdl = mdl_fp_load(fp);
	read_123(fp);

	Data_t * data = load_test_data(fp);
	read_123(fp);

	uint total = mdl->total;
	uint lines = data->lines;
	uint locds = mdl->locds;
	uint weights = mdl->weights;

	float * compare_array = load_float_array(lines * total, fp);
	read_123(fp);

	//	############ Use_t & Cpu_t #############

	//	============= Cpu_t =============

	printf("==========================================\n");
	printf("================= Cpu_t ==================\n");
	printf("==========================================\n");

	Cpu_t * cpu = (Cpu_t*)cpu_mk(mdl, data);

	cpu_set_input(cpu);
	cpu_forward(cpu);
		
	//	Compare
	if (test_package_compare_cpu_and_cpu(compare_array, cpu->var, mdl->total*data->lines)) {
		if (print_all)
			cpu_compare_vars(cpu, compare_array);
		OK("Cpu_t->var passed correctly.")
	} else {
		cpu_compare_vars(cpu, compare_array);
		ERR("Il y a des Erreures avec Cpu_t->var")
	}

	cpu_free(cpu);

	//	============= Use_t =============

	printf("==========================================\n");
	printf("================= Use_t ==================\n");
	printf("==========================================\n");

	Use_t * use = use_mk(mdl, data);

	use_set_input(use);
	use_forward(use);

	//	Compare
	if (test_package_compare_cpu_and_gpu(compare_array, use->var_d, mdl->total*data->lines)) {
		if (print_all)
			use_compare_vars(use, compare_array);
		OK("Use_t->var_d passed correctly.")
	} else {
		use_compare_vars(use, compare_array);
		ERR("Il y a des Erreures avec Use_t->var_d")
	}

	use_free(use);

	//  =================================

	free(compare_array);

	//	################ Train_t Involving ###############

	printf("==========================================\n");
	printf("================= Train_t ==================\n");
	printf("==========================================\n");

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
	
	//============= FORWARD ==============
	train_null_grad_meand(train);
	train_set_input(train);
	train_forward(train, 0);

	//		### WEIGHTS ###
	compare_array = load_float_array(sets * weights, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_weight, sets*weights)) {
		if (print_all)
			train_compare_weights(train, compare_array);
		OK("Train_t->_weight passed correctly.");
	} else {
		train_compare_weights(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_weight")
	}
	free(compare_array);
	read_123(fp);

	//		### VARS ###
	compare_array = load_float_array(lines * sets * total, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_var, lines*sets*total)) {
		if (print_all)
			train_compare_vars(train, compare_array);
		OK("Train_t->_var passed correctly.");
	} else {
		train_compare_vars(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_var")
	}
	free(compare_array);
	read_123(fp);

	//		### LOCDS ###
	compare_array = load_float_array(lines * sets * locds, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_locd, lines*sets*locds)) {
		if (print_all)
			train_compare_locds(train, compare_array);
		OK("Train_t->_locd passed correctly.");
	} else {
		train_compare_locds(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_locd")
	}
	free(compare_array);
	read_123(fp);

	//================ BACKWARD =============
	opti_dloss(opti_space);
	train_backward(train, 0);

	//		### GRADS ###
	compare_array = load_float_array(lines * sets * total, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_grad, lines*sets*total)) {
		if (print_all)
			train_compare_grads(train, compare_array);
		OK("Train_t->_grad passed correctly.");
	} else {
		train_compare_grads(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_grad")
	}
	free(compare_array);
	read_123(fp);

	//		### MEANDS ###
	compare_array = load_float_array(sets * weights, fp);
	if (test_package_compare_cpu_and_gpu(compare_array, train->_meand, sets * weights)) {
		if (print_all)
			train_compare_meands(train, compare_array);
		OK("Train_t->_meand passed correctly.");
	} else {
		train_compare_meands(train, compare_array);
		ERR("Il y a des Erreures avec Train_t->_meand")
	}
	free(compare_array);
	read_123(fp);

	//free
	opti_free(opti_space);
	train_free(train);
	data_free(data);
	mdl_free(mdl);
};
