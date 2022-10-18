#pragma once

//	This includes Global #define for both kernel and package headers
#include "package/meta.cuh"

//	This includes all the kernel headers
//	It includes optis.cuh that includes train.cuh ...
#include "kernel/head/testpackage.cuh"

//	This include all the package headers
#include "package/insts/insts.cuh"
#include "package/optis/optis.cuh"
#include "package/scores/scores.cuh"

//	Arrays are declared in headers and writed in package/src/*.cu