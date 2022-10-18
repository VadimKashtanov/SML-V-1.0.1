#pragma once

#include "kernel/head/train.cuh"

/*
	Vu qu'il y a des `if` pour verifier que c'est pas hors border, on pourrait optimiser.

	Le `if` n'existe que sur les bords. Il faudrait faire 2 fonctions.
	Une fonction `valide` qui calcule la convolution sans les `if`
	Et la partie `same` (celle qui est sous padding), avec que des `if`.

	(dans 2 dimention les images ont 4 coté (haut,gauche,bas,droit))
	(donc on peut eventuellement en 2D faire 4 fonctions)
	(et chaqu'une a un même paterne qui calcule sans `if`)
	(je réfléchissais en L1 comment faire en 2D un truc sans `if`)
	(c'est un peut complexe et un peut lourd pour pas grand chose)
	(eventuellement on peut eviter les `coins`, donc faire un approximation)
	(l'internet est surtout celui de la performance technique)
	(j'ai surement mis des commentaires et des bouts de code)
	(dans les anciennes version sur lequels j'ai bossé en Terminale/début L1)

*/

//Ax, Ay, K, n0, n1, stride, activ, drop_rate
//['Ax','Ay', 'Kx', 'Ky', 'n0', 'n1', 'strideX', 'strideY', 'paddingX', 'paddingY', 'activ', 'input_start','ystart','wstart','locdstart', 'drop_rate']

//	================== Use ==================

__global__
void kconvl2d_use_th1x1(
	uint Yx, uint Yy,
	uint n0, uint n1, uint Ax, uint Ay,
	uint Kx, uint Ky,
	uint strideX, uint strideY,
	uint paddingX, uint paddingY,
	uint activ,
	uint time,
	uint total, uint wsize,
	uint istart, uint wstart, uint ystart,
	float * var, float * weight);

//========================		Train_t	  =========================

//----------------------------- forward ---------------------------

__global__
void kconvl2d_forward_th1x1(
	uint Yx, uint Yy,
	uint n0, uint n1, uint Ax, uint Ay,
	uint Kx, uint Ky,
	uint strideX, uint strideY,
	uint paddingX, uint paddingY,
	uint activ,
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	uint seed, float drop_rate,
	uint set, uint sets,
	float * var, float * weight, float * locd);

//----------------------------- backward ---------------------------

__global__
void kconvl2d_backward_th1x1(
	uint Yx, uint Yy,
	uint n0, uint n1, uint Ax, uint Ay,
	uint Kx, uint Ky,
	uint strideX, uint strideY,
	uint paddingX, uint paddingY,
	uint activ,
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	uint seed, float drop_rate,
	uint set, uint sets,
	float * var, float * weight, float * locd,
	float * grad, float * meand);