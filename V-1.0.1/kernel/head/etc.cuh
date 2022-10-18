#pragma once

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <assert.h>

typedef uint32_t uint;

//	true == same
#define COMPARE_DEEPH 0.00001	//where 0.000001 befor
#define compare_floats(a, b, p) (fabs(a-b) < p) //((a - p) < b) && (a + p) > b

//int [0, umax(1 << 32)]  2^32 = (2^4)^8 = (0x10)^8 = 0x100000000
#define pseudo_randomi(seed) ((1234*(seed+1235))% 0x100000000)//((seed*seed*seed*7901) % 6997) / 6997

//float [0, 1]
//https://en.wikipedia.org/wiki/Pseudorandom_number_generator#Implementation
#define pseudo_randomf(seed) ((float)((1234*(seed+1235))%1000))/1000//(((float)(((seed*seed*seed+1)*7901) % 6997)) / 6997.0)
#define pseudo_randomf_minus1_1(seed) (2*pseudo_randomf(seed) - 1)

#define OK(str, ...) printf("[\033[35;1;41mOk\033[0m]:\033[96m%s:(%d)\033[32m: " str "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__);
#define MSG(str, ...) printf("[\033[35;1;41mWarrning\033[0m]:\033[96m%s:(%d)\033[35m: " str "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__);
#define ERR(str, ...) do {printf("[\033[30;101mError\033[0m]:\033[96m%s:(%d)\033[30m: " str "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__);raise(SIGINT);} while (0);

//	Definire 1 foix __constant__ est uniquement possible avec un .cuh
//sizeof(float) == 4 ; ( << 1) == *2; (1 << 16)/(1 << 2) == (1 << 14); max = 65536 bytes = 1 << 16 bytes = 16384 floats = 1 << 14 floats;
#define MAX_CONST_FLOATS 1 << 14
__constant__ float const_mem[1 << 14];

//	Shared memory
//<<<grid,block,shared_amount>>> amount will be alloc there. So kernel will have to use this single array to access the dynamicly allocaed __shared__ memory
extern __shared__ float dynamic__shared__[];	//no needs for size

//	Aide au deboguage
#define SAFE_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess)ERR("Cuda Error : %s", cudaGetErrorString(err));} while(0);

//	70 divise en thread de 32 == 3 block (dont le derniers qui n'utilisera pas tous les threads). 70 == 2*32 + 6 == 3*32 - 26
#define KERN_DIV(elements, thx) (((elements - elements%thx)/thx)+1)

//// Arguments parsing
void etc_parse_arguments(uint argc, char ** argv, uint paramc, const char ** paramv, char ** correspondance);	//paramv = {'out', 'sets', 'score'}
																												//./programme -out data.bin -sets 4 
																												//will give in correspondance : {'data.bin', '4', 0} 