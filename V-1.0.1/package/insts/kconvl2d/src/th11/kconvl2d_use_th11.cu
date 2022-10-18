#include "package/insts/kconvl2d/head/kconvl2d.cuh"

/*
	Each kernel compute one pixel for each channels (n1 pixels)

	<<<dim3(KERN_DIV(Yx,32), KERN_DIV(Yy,32), _n1), dim3(32,32, 1)>>>
*/

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
	float * var, float * weight)
{
	uint _Yx = threadIdx.x + blockIdx.x * blockDim.x;	//pour chaque pixel dans Input
	uint _Yy = threadIdx.y + blockIdx.y * blockDim.y;	//donc on va projeter ce pixel sur Input avec (x,y) 
	uint _n1 = blockIdx.z;

	//	Projection du Centre de la convolution (_Yx,Yy) sur input
	uint x = _Yx*strideX + paddingX;
	uint y = _Yy*strideY + paddingY;

	if (_Yx < Yx && _Yy < Yy) {

		uint _pixelpos, _kernelpos;

		uint ker_radiusX = (Kx-1)/2;
		uint ker_radiusY = (Ky-1)/2;

		float _sum = 0;

		uint start_ker_x = (x >= ker_radiusX) ? 0 : (ker_radiusX - x);	//max((ker_radius-x), 0)  it's kind of distance beetwin kernel border and image border
		uint start_ker_y = (y >= ker_radiusY) ? 0 : (ker_radiusY - y);

		uint end_ker_x = (x < (Ax - ker_radiusX)) ? Kx : (Kx - ((x+ker_radiusX) - Ax+1));
		uint end_ker_y = (y < (Ay - ker_radiusY)) ? Ky : (Ky - ((y+ker_radiusY) - Ay+1));

		for (uint _n0=0; _n0 < n0; _n0++) {
			for (uint ker_y=start_ker_y; ker_y < end_ker_y; ker_y++) {
				for (uint ker_x=start_ker_x; ker_x < end_ker_x; ker_x++) {
					_pixelpos = time*total + istart + _n0*Ax*Ay + (y+ker_y-ker_radiusY)*Ax + (x+ker_x-ker_radiusX);
					_kernelpos = wstart + _n1*Kx*Ky*n0 + _n0*Kx*Ky + (ker_y*Kx + ker_x);

					_sum += var[_pixelpos] * weight[_kernelpos];
				}
			}
		}

		_pixelpos = _n1*Yx*Yy + ((y-paddingY)/strideY)*Yx + ((x-paddingX)/strideX);
		_sum += weight[wstart + n1*n0*Kx*Ky + _pixelpos];

		if (activ == 0) _sum = 1 / (1 + exp(-_sum));
		else if (activ == 1) _sum = tanh(_sum);
		else if (activ == 2) _sum = exp(-_sum * _sum);
		else if (activ == 3) _sum = _sum * (_sum > 0);
		//else _tmp = tmp

		var[time*total + ystart + _pixelpos] = _sum;
	}
}