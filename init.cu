/*
 * init.cu
 *
 *  Created on: Aug 28, 2019
 *      Author: ramiro
 */

#include "struct.h"
#include "device_funcs.cuh"
#include "helper_math.h"
#include "const.h"
#include "params.h"
#include <stdio.h>



__device__ float sourceCoeff(int c, int r, int s, int id, int3 volSize,
		float4 params) {
	const int3 pos0 = { volSize.x / 2, volSize.y / 2, volSize.z / 2 };
	const float dx = c - pos0.x, dy = r - pos0.y, dz = s - pos0.z;
	const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
	const float dividend = (mc * amu2kg) * Q * fi * expf(-dist / lambda);
	const float divisor = 4.f * PI * awind * dist * dist
			* (rhowind * pow(mPerkm, 3));
	const float sourceCoeff = dividend / divisor;
	return __log10f(sourceCoeff) - params.x;
}


__device__ float lossCoeff(int c, int r, int s, int id, int3 volSize,
		float4 params) {
	const int3 pos0 = { volSize.x / 2, volSize.y / 2, volSize.z / 2 };
	const float dx = c - pos0.x, dy = r - pos0.y, dz = s - pos0.z;
	const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
	float etemp, edens, recomb;
	//electronic temperature and recombination rate calculation
	if (dist <= pow(10.f, 3.2f)) {
		etemp = 100.f;
		//printf("%f\n",etemp);
		if (etemp <= 200.f) {
			recomb = recomb0 * sqrtf(300.f / etemp);
		} else {
			recomb = 2.342f * recomb0
					* pow(etemp, 0.2553f - 0.1633f * __log10f(etemp));
		}
	}

	if (dist > pow(10.f, 3.2f) && dist <= pow(10.f, 3.84f)) {
		etemp = pow(10.f, 1.143f * __log10f(dist) - 1.667f);
		//  printf("%f\n",etemp);
		if (etemp <= 200.f) {
			recomb = recomb0 * sqrtf(300.f / etemp);
		} else {
			recomb = 2.342f * recomb0
					* pow(etemp, 0.2553f - 0.1633f * __log10f(etemp));
		}
	}
	if (dist > pow(10.f, 3.84f) && dist <= pow(10.f, 4.f)) {
		etemp = pow(10.f, 10.965f * __log10f(dist) - 39.3725f);
		if (etemp <= 200.f) {
			recomb = recomb0 * sqrtf(300.f / etemp);
		} else {
			recomb = 2.342f * recomb0
					* pow(etemp, 0.2553f - 0.1633f * __log10f(etemp));
		}
	}
	if (dist > pow(10.f, 4.f) && dist <= pow(10.f, 5.f)) {
		etemp = pow(10.f, 0.5135f * __log10f(dist) + 2.4325f);
		//printf("%f\n",etemp);
		if (etemp <= 200.f) {
			recomb = recomb0 * sqrtf(300.f / etemp);
		} else {
			recomb = 2.342f * recomb0
					* pow(etemp, 0.2553f - 0.1633f * __log10f(etemp));
		}
	}
	if (dist >= pow(10.f, 4.f)) {
		etemp = 1.e+5f;
		if (etemp <= 200.f) {
			recomb = recomb0 * sqrtf(300.f / etemp);
		} else {
			recomb = 2.342f * recomb0
					* pow(etemp, 0.2553f - 0.1633f * __log10f(etemp));
		}
	}
	edens = ((nsw * pow(km2cm, 3)) * msw) / mc;
	return recomb * (lambda / awind) * edens;
}





/*__global__
 void InitEuler(float *gas_density, float *gas_vx, float *gas_vy, float *gas_vz){

 }*/

/*__global__
 void initialization(float *gas_density, float *gas_vx,
 float *gas_vy, float *gas_vz){
 InitEuler(gas_density, gasvx, gasvy, gasvz);
 }*/
