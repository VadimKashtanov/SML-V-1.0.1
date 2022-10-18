#include "kernel/head/testpackage.cuh"

float * load_float_array(uint len, FILE * fp) {
	float * ret = (float*)malloc(sizeof(float) * len);
	fread(ret, sizeof(float), len, fp);
	return ret;
};

void read_123(FILE * fp) {
	//Read next `uint` and check if it's 123.
	//if not, there is a probleme in reading. Maybe read to much or not enougth

	uint _tmp;
	fread(&_tmp, sizeof(uint), 1, fp);
	if (_tmp != 123)
		ERR("Expected an 123 number, but get : %i", 123);
};

//=====================================================================

static bool compare_arrays(float * cpu0, float * cpu1, uint count)
{
	for (uint i=0; i < count; i++) {
		if (compare_floats(cpu0[i], cpu1[i], 0.0001) != true) {
			return false;
		}
	}
	return true;
};

bool test_package_compare_cpu_and_gpu(float * cpu0, float * gpu_d, uint count)
{
	float * cpu = (float*)malloc(sizeof(float) * count);
	SAFE_CUDA(cudaMemcpy(cpu, gpu_d, sizeof(float) * count, cudaMemcpyDeviceToHost));
	bool ret = compare_arrays(cpu0, cpu, count);
	free(cpu);
	return ret;
};

bool test_package_compare_cpu_and_cpu(float * cpu0, float * cpu1, uint count)
{
	return compare_arrays(cpu0, cpu1, count);
};

//==========================================================================

Data_t * load_test_data(FILE * fp)
{
	uint batchs, lines, inputs, outputs;

	fread(&batchs, sizeof(uint), 1, fp);
	fread(&lines, sizeof(uint), 1, fp);
	fread(&inputs, sizeof(uint), 1, fp);
	fread(&outputs, sizeof(uint), 1, fp);

	Data_t * ret = data_load(batchs, inputs, outputs, lines);

	data_cudmalloc(ret);

	fread(ret->input, sizeof(float), lines*inputs, fp);
	fread(ret->output, sizeof(float), lines*outputs, fp);

	data_cudamemcpy(ret);

	return ret;
};