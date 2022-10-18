#include "package/package.cuh"

//./print_data data.bin 5
//		va ecrire le batch numero 5

int main(int argc, char ** argv) {
	if (argc != 3)
		ERR("Il faut donner l'addresse du fuchier et le numéro du batch en argument")

	uint batch_id = atoi(argv[2]);

	Data_t * data = data_open(argv[1]);

	FILE * fp = fopen(argv[1], "rb");
	data_load_batch(data, fp, batch_id);
	fclose(fp);

	data_print_info(data);

	printf("\033[90m Batch %i\033[0m \n", batch_id);
	data_print_batch(data);

	data_free(data);
};