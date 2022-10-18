#include "package/insts/lstm2d/head/lstm2d.cuh"

/*
	Reecrire completement LSTM

	Mettre dans la classe des fonctions complexes

	Le méta-model genetic doit arriver lui meme avec `+`, `dot1d`, 'activ' ... a ça.

	On optimisera apres.

*/


/*			  =======
			  |     |
			  |		|
			  |	.W	|
			  |		|
			  |		|
			  =======
============= =======
|	.input	| | 	|	input@W
============= =======
				 +
			  =======
			  |	.U  |
			  |	    |
			  =======
	  ======= =======
	  |h[-1]| |		|  h[-1]@U
	  ======= =======
				 +
			  =======
			  |	.B	|
			  =======
*/

/*	We could use atomicAdd with 1 direct backward function

*/

/*
__global__  //ca veut dire que le kernel est position sur les cooredonne de l'input, et chaque kernel est associe a un pixel de l'input. Puis on backward on ligne verticale de .W
void lstm2d_backward_INPUT_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//	Backward grad(input)
	//	meand(.W) of f0,f1,f2,g0
	

	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	uint ipos = total*sets*time + total*set + istart + Bx*Ax + (y*Ax + x);

	//input = Ax*Ay, and the (x,y) pixel is in input. Then we backward .W and this pixel gradient
	if (x < Ax && y < Ay && pseudo_randomf(seed + ipos) > drop_rate) {	//if input[x] is droped, following will be *0

		float grad_input_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		//uint W = wsize*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		//uint vpos = total*sets*time + set*total + istart + x;
		float xvalue = var[ipos];

		float e;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos, outpos;	//wpos   = position du weight en question
												//epos,e_1pos,hpos = output `e` ou `h` (car output = `e` + `h`). e_1 est e[-1]
												//outpos = (y*Bx+k) juste pour calculer de quel pixel de Y nous prenon le locd (car on backward chaque colone de output mais les weights d'une meme ligne) 

		//	Backward W
		for (uint k=0; k < Bx; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y

			outpos = y*Bx + k;

			epos = out + outpos;
			e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
			hpos = out + Bx*Ay + outpos;

			e = var[epos];

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
			f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
			f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
			g0 = locd[locdpos + 3*Bx*Ay + outpos];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			//if time > 0:
			grad[e_1pos] += de * f0;		//we can't store only 4 locds, because how will we get de*f0 ?

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = wsize*set + wstart + 0*lineW + (k*Bx + y);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf0 * xvalue;
			atomicAdd(meand + wpos, dsf0 * xvalue);
			grad_input_compute += dsf0 * weight[wpos];

			//	f1
			wpos = wsize*set + wstart + 1*lineW + (k*Bx + y);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf1 * xvalue;
			atomicAdd(meand + wpos, dsf1 * xvalue);
			grad_input_compute += dsf1 * weight[wpos];

			//	f2
			wpos = wsize*set + wstart + 2*lineW + (k*Bx + y);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf2 * xvalue;
			atomicAdd(meand + wpos, dsf2 * xvalue);
			grad_input_compute += dsf2 * weight[wpos];

			//	g0
			wpos = wsize*set + wstart + 3*lineW + (k*Bx + y);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsg0 * xvalue;
			atomicAdd(meand + wpos, dsg0 * xvalue);
			grad_input_compute += dsg0 * weight[wpos];
		}

		//	Backward input
		grad[ipos] += grad_input_compute;
		//atomicAdd(grad + vpos, grad_input_compute);
	}
}

__global__
void lstm2d_backward_H1_BIAS_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint x = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (x < Bx && y < Ay) {	//Only input is under drop. h is an output. It's values, could be droped, but in an other instruction

		float grad_H1_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		uint h1pos = total*sets*(time-1) + set*total + istart + (y*Bx + x);	//h[-1] pos
		float h1val = var[h1pos];

		uint W = wsize*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		//float _grad;	//of h[t]

		//float chain_deriv;
		float e;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos, outpos;	//wpos   = position du weight en question
												//epos,e_1pos,hpos = output `e` ou `h` (car output = `e` + `h`). e_1 est e[-1]
												//outpos = (y*Bx+k) juste pour calculer de quel pixel de Y nous prenon le locd (car on backward chaque colone de output mais les weights d'une meme ligne) 

		for (uint k=0; k < Bx; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y
										//en fait k est la colone de la matrice. la ligne est `y` du kernel
										//et le `x` du kernel determine le pixel `h[-1]` et la ligne dans .U 

			outpos = y*Bx + k;
			
			epos = out + outpos;
			e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
			hpos = out + Bx*Ay + outpos;

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
			f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
			f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
			g0 = locd[locdpos + 3*Bx*Ay + outpos];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			e = var[epos];

			//if time > 0:
			grad[e_1pos] += de * f0;

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = W + 0*lineW + _W + (x*Bx + k);					//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf0 * h1val;
			atomicAdd(meand + wpos, dsf0 * h1val);
			grad_H1_compute += dsf0 * weight[wpos];

			//	f1
			wpos = W + 1*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf1 * h1val;
			atomicAdd(meand + wpos, dsf1 * h1val);
			grad_H1_compute += dsf1 * weight[wpos];

			//	f2
			wpos = W + 2*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf2 * h1val;
			atomicAdd(meand + wpos, dsf2 * h1val);
			grad_H1_compute += dsf2 * weight[wpos];
		
			//	g0
			wpos = W + 3*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsg0 * h1val;
			atomicAdd(meand + wpos, dsg0 * h1val);
			grad_H1_compute += dsg0 * weight[wpos];
		}

		//	Backward h[-1]
		grad[h1pos] += grad_H1_compute;
		//atomicAdd(grad + vpos, grad_input_compute);

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Bx,Ay>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//

		outpos = y*Bx + x;

		epos = out + outpos;
		e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
		hpos = out + Bx*Ay + outpos;

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
		f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
		f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
		g0 = locd[locdpos + 3*Bx*Ay + outpos];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[W + 0*lineW + _W + _U + (y*Bx + x)] += dsf0;

		//	f1
		meand[W + 1*lineW + _W + _U + (y*Bx + x)] += dsf1;

		//	f2
		meand[W + 2*lineW + _W + _U + (y*Bx + x)] += dsf2;

		//	g0
		meand[W + 3*lineW + _W + _U + (y*Bx + x)] += dsg0;
	}
};

__global__
void lstm2d_backward_BIAS_ONLY_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint x = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (x < Bx && y < Ay) {	//Only input is under drop. h is an output. It's values, could be droped, but in an other instruction

		uint W = wsize*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		//float _grad;	//of h[t]

		//float chain_deriv;
		float e;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint epos, e_1pos, hpos, outpos;	//wpos   = position du weight en question
												//epos,e_1pos,hpos = output `e` ou `h` (car output = `e` + `h`). e_1 est e[-1]
												//outpos = (y*Bx+k) juste pour calculer de quel pixel de Y nous prenon le locd (car on backward chaque colone de output mais les weights d'une meme ligne) 

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Bx,Ay>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//
		
		outpos = y*Bx + x;

		epos = out + outpos;
		e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
		hpos = out + Bx*Ay + outpos;

		e = var[epos];

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
		f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
		f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
		g0 = locd[locdpos + 3*Bx*Ay + outpos];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[W + 0*lineW + _W + _U + (y*Bx + x)] += dsf0;

		//	f1
		meand[W + 1*lineW + _W + _U + (y*Bx + x)] += dsf1;

		//	f2
		meand[W + 2*lineW + _W + _U + (y*Bx + x)] += dsf2;

		//	g0
		meand[W + 3*lineW + _W + _U + (y*Bx + x)] += dsg0;
	}
};
*/

__global__
void lstm2d_backward_th11(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	uint inp = total*sets*time + total*set + istart;
	uint W = wsize*set + wstart;
	uint out = total*sets*time + total*set + ystart;
	uint locdpos = lsize*sets*time + lsize*set + lstart;

	uint _W = Bx * Ax;
	uint _U = Bx * Bx;
	uint _B = Bx * Ay;
	
	uint lineW = _W + _U + _B;

	float f0=0,f1=0,f2=0,g0=0;

	float e, dH, de;
	float dsf0, dsf1, dsf2, dsg0;

	uint epos, e_1pos, hpos, vpos, wpos;
	
	if (x < Bx && y < Ay)
	{
		epos = out + (y*Bx + x);
		e_1pos = total*sets*(time-1) + total*set + ystart + (y*Bx + x); //if l == 0 , e_1pos <= 0
		hpos = out + Bx*Ay + (y*Bx + x);	//Bx*Ay is the space of `e`

		e = var[epos];
		dH = grad[hpos];
	
		f0 = locd[locdpos + 0*Bx*Ay + (y*Bx + x)] * dH;
		f1 = locd[locdpos + 1*Bx*Ay + (y*Bx + x)] * dH;
		f2 = locd[locdpos + 2*Bx*Ay + (y*Bx + x)] * dH;
		g0 = locd[locdpos + 3*Bx*Ay + (y*Bx + x)] * dH;
	
		de = grad[epos] + dH * f2;	//		#grad(e) += dH*f2
	
		grad[epos] = de;
	
		if (time > 0) {
			atomicAdd(&grad[e_1pos], de * f0);
		}
		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);
	
		//// .W
		for (uint k=0; k < Ax; k++) {
			vpos = inp + y*Ax + k;

			if (pseudo_randomf(seed + vpos) > drate) {
				wpos = (k*Bx + x);
		
				//f0 += var[vpos]*w[W + wpos]
				atomicAdd(&grad[vpos], dsf0 * weight[W + 0*lineW + wpos]);
				atomicAdd(&meand[W + 0*lineW + wpos], dsf0 * var[vpos]);
		
				//f1 += var[vpos]*w[W + lineW + wpos]
				atomicAdd(&grad[vpos], dsf1 * weight[W + 1*lineW + wpos]);
				atomicAdd(&meand[W + 1*lineW + wpos], dsf1 * var[vpos]);
		
				//f2 += var[vpos]*w[W + 2*lineW + wpos]
				atomicAdd(&grad[vpos], dsf2 * weight[W + 2*lineW + wpos]);
				atomicAdd(&meand[W + 2*lineW + wpos], dsf2 * var[vpos]);
		
				//g0 += var[vpos]*w[W + 3*lineW + wpos]
				atomicAdd(&grad[vpos], dsg0 * weight[W + 3*lineW + wpos]);
				atomicAdd(&meand[W + 3*lineW + wpos], dsg0 * var[vpos]);
			}
		}
	
		//// .U
		if (time > 0) {
			for (uint k=0; k < Bx; k++) {
				//out == t
				//out - total*sets == sets*total*(l-1) + _set*total + istart
				vpos = sets*total*(time-1) + set*total + ystart + Bx*Ax + (y*Ax + k);// 	#h[-1][y][x]
				wpos = _W + (k*Bx + y);
	
				//f0 += var[vpos]*w[W + wpos]
				atomicAdd(&grad[vpos], dsf0 * weight[W + 0*lineW + wpos]);
				atomicAdd(&meand[W + 0*lineW + wpos], dsf0 * var[vpos]);
	
				//f1 += var[vpos]*w[W + lineW + wpos]
				atomicAdd(&grad[vpos], dsf1 * weight[W + 1*lineW + wpos]);
				atomicAdd(&meand[W + 1*lineW + wpos], dsf1 * var[vpos]);
	
				//f2 += var[vpos]*w[W + 2*lineW + wpos]
				atomicAdd(&grad[vpos], dsf2 * weight[W + 2*lineW + wpos]);
				atomicAdd(&meand[W + 2*lineW + wpos], dsf2 * var[vpos]);
	
				//g0 += var[vpos]*w[W + 3*lineW + wpos]
				atomicAdd(&grad[vpos], dsg0 * weight[W + 3*lineW + wpos]);
				atomicAdd(&meand[W + 3*lineW + wpos], dsg0 * var[vpos]);
			}
		}

		//// .B
		wpos = _W + _U + (y*Bx + x);
	
		atomicAdd(&meand[W + 0*lineW + wpos], dsf0);
		atomicAdd(&meand[W + 1*lineW + wpos], dsf1);
		atomicAdd(&meand[W + 2*lineW + wpos], dsf2);
		atomicAdd(&meand[W + 3*lineW + wpos], dsg0);
	}
}