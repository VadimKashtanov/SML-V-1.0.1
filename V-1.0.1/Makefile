#
#	Gcc tools :
#		gdb, 		valgind
#		cuda-gdb
#		
#	Je peux utiliser `nvprof`
#
#	cuda-gdb ./<program>
#	valgrind --track-origins=yes ./<package>           (eventuellement avec --leak-check=full, mais les arguments sont toujours avant le programe)

#device (debbug les __global__ kernel)
	DEBBUG = -G -O0 # -Mcuda=debug
#host (debbug les fonctions classiques sur le CPU)
	#DEBBUG = -g -O0 #-Mcuda=debug
#optimized out
	#DEBBUG = -O3

#			-I$(0) est le package
ARGS = $(DEBBUG) -lm -I. --compiler-options -Wall

all: test_package optimize_mdl print_mdl print_data test_mdl print_line_format compare_mdl_outputs

*.o:
	@ printf "[\033[35;1;41m***\033[0m] ================= KERNEL ===================\n"
	nvcc -c $(ARGS) $(shell find kernel -type f -name "*.cu") $(shell find package/insts -type f -name "*.cu") $(shell find package/optis -type f -name "*.cu") $(shell find package/scores -type f -name "*.cu")

##
##	Les programmes sont les suivants
##
##		test_package			|	Tester toutes les instructions, les optimizeurs, les ...
##		optimize_mdl			|	Optimiser un model a partire d'un Data file. C'est tout.
##		print_mdl				|	Just show mdl
##		print_data				|	Print data
##		test_mdl				|	Forward a batch with Cpu_t
##		print_line_format		|	Print data line in 2dsquare of histogram
##		compare_mdl_outputs		|	Compare Outputs of 2 models in compare_mdl_outputs/compare.py
##

## 	Args : (name:str, path)
define generate_file
	@ printf "[\033[35;1;41m***\033[0m] ============= PROGRAM : $(1) ============\n"
	nvcc $(ARGS) *.o $(shell find $(2) -type f -name "*.cu") -o $(3)
endef

test_package: *.o
	$(call generate_file,"TEST_PACKAGE",package/programs/test_package,$@)

optimize_mdl: *.o
	$(call generate_file,"OPTIMIZE_MODEL",package/programs/optimize_mdl,$@)

print_mdl: *.o
	$(call generate_file,"PRINT_MDL",package/programs/print_mdl,$@)

print_data: *.o
	$(call generate_file,"PRINT_DATA",package/programs/print_data,$@)

test_mdl: *.o
	$(call generate_file,"TEST_MDL",package/programs/test_mdl,$@)

print_line_format: *.o
	$(call generate_file,"PRINT_LINE_FORMAT",package/programs/print_line_format,$@)

compare_mdl_outputs: *.o
	$(call generate_file,"COMPARE_MDL_OUTPUS",package/programs/compare_mdl_outputs,$@)