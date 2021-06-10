/*
 * kernel.cu
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#include "struct.h"
#include "kernel.h"

#include "device_funcs.cuh"
#include "flux_interface.cuh"

#include "helper_math.h"
#include <math.h>
#include <math_constants.h>
#include "const.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>


//////////////////////////////////////////////////////////////////
//   K E R N E L    F U N C T I O N S
__global__
void renderKernel(uchar4 *d_out, double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz, double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, float time, int w, int h,
		int3 volSize,	int method, float zs, float theta,
		float threshold, float dis, int id) {

	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = c + r * w;
	if ((c >= w) || (r >= h))
		return; // Check if within image bounds
	const uchar4 background = { 64, 0, 128, 0 };
	float3 source = { 0.f, 0.f, -zs };
	float3 pix = scrIdxToPos(c, r, w, h, 2 * volSize.z - zs);

	// apply viewing transformation: here rotate about y-axis
	source = yRotate(source, theta);
	pix = yRotate(pix, theta);

	// prepare inputs for ray-box intersection
	float t0, t1;
	const Ray pixRay = { source, pix - source };
	float3 center = { volSize.x / 2.f, volSize.y / 2.f, volSize.z / 2.f };
	const float3 boxmin = -center;
	const float3 boxmax = { volSize.x - center.x, volSize.y - center.y,
			volSize.z - center.z };

	// perform ray-box intersection test
	const bool hitBox = intersectBox(pixRay, boxmin, boxmax, &t0, &t1);

	//uchar4 shade;
	uchar4 shade = make_uchar4(96, 0, 192, 0); // background value
	if (!hitBox)
		shade = background; //miss box => background color
	else {
		if (t0 < 0.0f)
			t0 = 0.f; // clamp to 0 to avoid looking backward
		// bounded by points where the ray enters and leaves the box
		const Ray boxRay = { paramRay(pixRay, t0), paramRay(pixRay, t1)
				- paramRay(pixRay, t0) };
		if (method == 1)
			shade = sliceShader(d_dens, d_momPx, d_momPy, d_momPz, d_Bx,
					d_By, d_Bz, d_e, volSize, boxRay, threshold, dis,
					source, id);
		/*else if (method == 2)
			shade = rayCastShader(d_dens, d_momP, d_B, d_e, volSize, boxRay, threshold, id);
		else
			shade = volumeRenderShader(d_dens, d_momP, d_B, d_e, volSize, boxRay, threshold,
			NUMSTEPS, id);*/
	}
	d_out[i] = shade;
	__syncthreads();

	// Color bar
	if ((r <= 40) && (c >= (600/2)-128) && (c <= (600/2)+127)) {
		d_out[i].x = 255 - (c - (600/2)-128);
		d_out[i].y = 192;
		d_out[i].z = (c - (600/2)-128);
		d_out[i].w = 255;
	   // return;
	}

	// Establish the bar borders
	if (((r <= 41) && (c == (600/2)-129)) |
			((r <= 41) && (c == (600/2)+128)) |
			((r == 41) && (c >= (600/2)-128) && (c <= (600/2)+127))) {
			d_out[i].x = 0;
			d_out[i].y = 0;
			d_out[i].z = 0;
			d_out[i].w = 255;
		    return;
	}

	// Establish the bar scale for each physical quantity
	int iSegm, nSegm;
	switch (id) {
	  case 0: // mass density
		  nSegm = 6;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 1: // (mean) linear momentum
		  nSegm = 5;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 2: // (mean) magnetic field
		  nSegm = 6;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 3: // internal energy density
		  nSegm = 6;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 4: // gyroradius
		  nSegm = 5;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 5: // x-velocity component
		  nSegm = 1;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	  case 6: // ENAs production
		  nSegm = 10;
		  for ( iSegm = 1; iSegm < nSegm; iSegm++){
			  if ((r<=41) && (r>=30) && (c == (600/2)-129 + roundf(256/nSegm*iSegm))){
				  d_out[i].x = 0;
				  d_out[i].y = 0;
				  d_out[i].z = 0;
				  d_out[i].w = 255;
				  return;
			  }
		  }
		  break;
	}
}


__global__ void resetKernel(double *d_dens, double *d_momPx, double *d_momPy, double *d_momPz,
		double *d_Bx, double *d_By,double *d_Bz, double *d_e, int3 volSize, int id){
	const int w = volSize.x, h = volSize.y, d = volSize.z;
	const int col = blockIdx.x * blockDim.x + threadIdx.x; // column
	const int row = blockIdx.y * blockDim.y + threadIdx.y; // row
	const int stk = blockIdx.z * blockDim.z + threadIdx.z; // stack
	const int idx = col + (row * w) + (stk * w * h);
	if ((col >= w) || (row >= h) || (stk >= d)) return;

	//interplanetary initial conditions
//	const float initDens = ionDens(dist);
	const Vec3d v = {0.f, 0.f, 0.f};		//plasma speed[km/s]
	Vec3d xyz = {scale(col, w), scale(row, h), scale(stk, d)};
	double dist = sqrtf(dotProd(xyz, xyz));
	const float initDens = rhowind;			//plasma density[kg/m^3]
	const Vec3d B = {IMF, IMF, 0.f};		//Magnetic field [nT]

	// Normalization of physical quantities
	const float densN = densNorm(initDens);
	const Vec3d vN = vNorm(v);				//Velocity normalization (ion = neutral)
	const float pN = (initDens)*Boltz*eTemp(dist)/(mc*amu2kg)/
				(rhowind*awind*awind*powf(mPerkm,2));//Pressure normalization
	const Vec3d BN =BNorm(B);				//Magnetic field normalization

	// Copy the normalized physical quantities to arrays
	d_dens[idx] = densN;
	d_momPx[idx] = densN*vN.x;
	d_momPy[idx] = densN*vN.y;
	d_momPz[idx] = densN*vN.z;
	d_Bx[idx] = BN.x;
	d_By[idx] = BN.y;
	d_Bz[idx] = BN.z;
	d_e[idx] = 0.5f*(densN*dotProd(vN,vN) + 2.f*pN/(gamma-1.f) + dotProd(BN,BN));
	__syncthreads();
	//printf("%f\n",d_momPx[0]);
}

__global__ void evolutiveKernel(double *d_dens, double *d_momPx, double *d_momPy, double *d_momPz,
		double *d_Bx, double *d_By, double *d_Bz, double *d_e,
		int time, int3 volSize, int id){
//	extern __shared__ double s_dens[], s_momPx[], s_momPy[], s_momPz[],
//	s_Bx[], s_By[], s_Bz[], s_e[];
	// Global indices
	const int w = volSize.x, h = volSize.y, d = volSize.z;
	const int col = blockIdx.x * blockDim.x + threadIdx.x; // column
	const int row = blockIdx.y * blockDim.y + threadIdx.y; // row
	const int stk = blockIdx.z * blockDim.z + threadIdx.z; // stack
	if ((col >= w) || (row >= h) || (stk >= d)) return;
	const int idx = flattenIdx(col, row, stk, w, h, d);

	if ((row == 0) || (col == 0) || (row == w-1) || (col == h-1)){
//		d_dens[idx] = 1.f;
//		d_momPx[idx]=10.f;
//		d_momPy[idx]=0.f;
//		d_momPz[idx]=0.f;
//		const Vec3d B = {IMF, IMF, 0.f};		//Magnetic field [nT]
//		const Vec3d BN =BNorm(B);				//Magnetic field normalization
//		d_Bx[idx] = BN.x;
//		d_By[idx] = BN.y;
//		d_Bz[idx] = BN.z;
		return;
	}

//	if (row == 0){
//		d_momPx[idx]=10.f;
//		return;
//	}

	//d_dens[flattenIdx(col, row, stk, w, h, d)] = densN;

	//if (idx==1000000) printf("%f\n", d_dens[idx]);
	// Local width and height
/*	const int s_w = blockDim.x + 2 * RAD;
	const int s_h = blockDim.y + 2 * RAD;
	const int s_d = blockDim.z + 2 * RAD;
	// Local indices
	const int s_col = threadIdx.x + RAD;
	const int s_row = threadIdx.y + RAD;
	const int s_stk = threadIdx.z + RAD;
	const int s_idx = flattenIdx(s_col, s_row, s_stk, s_w, s_h, s_d);
	// Load regular cells
	s_dens[s_idx] = d_dens[idx];
	s_momPx[s_idx] = d_momPx[idx];
	s_momPy[s_idx] = d_momPy[idx];
	s_momPz[s_idx] = d_momPz[idx];
	s_Bx[s_idx] = d_Bx[idx];
	s_By[s_idx] = d_By[idx];
	s_Bz[s_idx] = d_Bz[idx];
	s_e[s_idx] = d_e[idx];
	// Load halo cells
	if (threadIdx.x < RAD) {
	    s_dens[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_dens[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_momPx[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_momPx[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_momPy[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_momPy[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_momPz[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_momPz[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_Bx[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_Bx[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_By[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_By[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_Bz[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_Bz[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	    s_e[flattenIdx(s_col - RAD, s_row, s_stk, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col - RAD, row, stk, w, h, d)];
	    s_e[flattenIdx(s_col + blockDim.x, s_row, s_stk, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col + blockDim.x, row, stk, w, h, d)];
	}
	if (threadIdx.y < RAD) {
	    s_dens[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_dens[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_momPx[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_momPx[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_momPy[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_momPy[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_momPz[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_momPz[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_Bx[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_Bx[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_By[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_By[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_Bz[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_Bz[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	    s_e[flattenIdx(s_col, s_row - RAD, s_stk, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col, row - RAD, stk, w, h, d)];
	    s_e[flattenIdx(s_col, s_row + blockDim.y, s_stk, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col, row + blockDim.y, stk, w, h, d)];
	}
	if (threadIdx.z < RAD) {
	    s_dens[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_dens[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_dens[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_momPx[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_momPx[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_momPx[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_momPy[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_momPy[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_momPy[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_momPz[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_momPz[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_momPz[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_Bx[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_Bx[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_Bx[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_By[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_By[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_By[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_Bz[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_Bz[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_Bz[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	    s_e[flattenIdx(s_col, s_row, s_stk - RAD, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col, row, stk - RAD, w, h, d)];
	    s_e[flattenIdx(s_col, s_row, s_stk + blockDim.z, s_w, s_h, s_d)] =
	      d_e[flattenIdx(col, row, stk + blockDim.z, w, h, d)];
	}

	if ((threadIdx.x < RAD) && (threadIdx.y < RAD) && (threadIdx.z < RAD)){
		s_dens[flattenIdx(s_col - RAD, s_row - RAD, s_stk - RAD, s_w, s_h, s_d)] =
				d_dens[flattenIdx(col - RAD, row - RAD, stk - RAD, w, h, d)];
		s_dens[flattenIdx(s_col - RAD, s_row - RAD, s_stk + blockDim.z, s_w, s_h, s_d)] =
			    d_dens[flattenIdx(col - RAD, row - RAD, stk + blockDim.z, w, h, d)];
		s_dens[flattenIdx(s_col - RAD, s_row + blockDim.y, s_stk + blockDim.z, s_w, s_h, s_d)] =
			    d_dens[flattenIdx(col - RAD, row + blockDim.y, stk + blockDim.z, w, h, d)];
		s_dens[flattenIdx(s_col + blockDim.x, s_row - RAD, s_stk - RAD, s_w, s_h, s_d)] =
			    d_dens[flattenIdx(col + blockDim.x, row - RAD, stk - RAD, w, h, d)];
		s_dens[flattenIdx(s_col + blockDim.x, s_row + blockDim.y, s_stk - RAD, s_w, s_h, s_d)] =
			    d_dens[flattenIdx(col + blockDim.x, row + blockDim.y, stk - RAD, w, h, d)];
		s_dens[flattenIdx(s_col + blockDim.x, s_row - RAD, s_stk + blockDim.z, s_w, s_h, s_d)] =
				d_dens[flattenIdx(col + blockDim.x, row - RAD, stk + blockDim.z, w, h, d)];
		s_dens[flattenIdx(s_col - RAD, s_row + blockDim.y, s_stk - RAD, s_w, s_h, s_d)] =
				d_dens[flattenIdx(col - RAD, row + blockDim.y, stk - RAD, w, h, d)];
		s_dens[flattenIdx(s_col + blockDim.x, s_row + blockDim.y, s_stk + blockDim.z, s_w, s_h, s_d)] =
				d_dens[flattenIdx(col + blockDim.x, row + blockDim.y, stk + blockDim.z, w, h, d)];
		s_momPx[flattenIdx(s_col - RAD, s_row - RAD, s_stk - RAD, s_w, s_h, s_d)] =
				d_momPx[flattenIdx(col - RAD, row - RAD, stk - RAD, w, h, d)];
		s_momPx[flattenIdx(s_col - RAD, s_row - RAD, s_stk + blockDim.z, s_w, s_h, s_d)] =
			    d_momPx[flattenIdx(col - RAD, row - RAD, stk + blockDim.z, w, h, d)];
		s_momPx[flattenIdx(s_col - RAD, s_row + blockDim.y, s_stk + blockDim.z, s_w, s_h, s_d)] =
			    d_momPx[flattenIdx(col - RAD, row + blockDim.y, stk + blockDim.z, w, h, d)];
		s_momPx[flattenIdx(s_col + blockDim.x, s_row - RAD, s_stk - RAD, s_w, s_h, s_d)] =
			    d_momPx[flattenIdx(col + blockDim.x, row - RAD, stk - RAD, w, h, d)];
		s_momPx[flattenIdx(s_col + blockDim.x, s_row + blockDim.y, s_stk - RAD, s_w, s_h, s_d)] =
			    d_momPx[flattenIdx(col + blockDim.x, row + blockDim.y, stk - RAD, w, h, d)];
		s_momPx[flattenIdx(s_col + blockDim.x, s_row - RAD, s_stk + blockDim.z, s_w, s_h, s_d)] =
				d_momPx[flattenIdx(col + blockDim.x, row - RAD, stk + blockDim.z, w, h, d)];
		s_momPx[flattenIdx(s_col - RAD, s_row + blockDim.y, s_stk - RAD, s_w, s_h, s_d)] =
				d_momPx[flattenIdx(col - RAD, row + blockDim.y, stk - RAD, w, h, d)];
		s_momPx[flattenIdx(s_col + blockDim.x, s_row + blockDim.y, s_stk + blockDim.z, s_w, s_h, s_d)] =
				d_momPx[flattenIdx(col + blockDim.x, row + blockDim.y, stk + blockDim.z, w, h, d)];
	}


*/
//	__syncthreads();

	// Scale threads to distance
	Vec3d xyz = {scale(col, w), scale(row, h), scale(stk, d)};
	double dist = sqrtf(dotProd(xyz, xyz));

/*	if ((row == 0) && (col <= w-1)){  //|| (row == 0) || (stk == 0)){
		d_dens[idx] = 1.f;
		d_momPx[idx] = 10.f;
		d_momPy[idx] = 10.f;
		d_momPz[idx] = 10.f;
		d_Bx[idx] = (IMF*0.7071067812f)*1.e-9f/sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2));
		d_By[idx] = (-IMF*0.7071067812f)*1.e-9f/sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2));
		d_Bz[idx] = 0.f;
		d_e[idx] = 0.5f*(300.f + (d_Bx[idx]*d_Bx[idx] + d_Bx[idx]*d_Bx[idx]));
//		d_e[idx] = 0.5f*(densN*dotProd(vN,vN) + 2.f*pN/(gamma-1.f) + dotProd(BN,BN));
		//printf("%.30f\n", d_Bx[idx]);
		return;
	}*/
/*	if ((row == h-1) && (col <= w-1)){
		d_dens[idx]=0.0;
		return;
	}*/

	//
	//double initDens = ionDens(dist);
	//double densN = densNorm(initDens);	//Density normalization

	const float volCell = ((2.f*LEN)/NX)*((2.f*LEN)/NY)*((2.f*LEN)/NZ);
	const float areaFluxX = volCell/((2.f*LEN)/NX);
	const float areaFluxY = volCell/((2.f*LEN)/NY);
	const float areaFluxZ = volCell/((2.f*LEN)/NZ);

	// Physical parameters in the comet position
//	if (dist <= LEN/50.f){
/*		d_dens[idx]=0.f;
		d_momPx[idx]=0.f;
		d_momPy[idx]=0.f;
		d_momPz[idx]=0.f;
		d_Bx[idx]=0.f;
		d_By[idx]=0.f;
		d_Bz[idx]=0.f;
		d_e[idx]=0.f;*/
//		return;
//	}

	// Return the boundaries as initial conditions
/*	if ((row == 0) || (col == 0) || (stk == 0) ||
			(row == w-1) || (col == h-1) || (stk == d-1)){
		// dens case: first and last cells are equal to the second and last-1 cells
		d_dens[flattenIdx(0, row, stk, w, h, d)] = d_dens[flattenIdx(1, row, stk, w, h, d)];
		d_dens[flattenIdx(col, 0, stk, w, h, d)] = d_dens[flattenIdx(col, 1, stk, w, h, d)];
		d_dens[flattenIdx(col, row, 0, w, h, d)] = d_dens[flattenIdx(col, row, 1, w, h, d)];
		d_dens[flattenIdx(col-1, row, stk, w, h, d)] = d_dens[flattenIdx(col-2, row, stk, w, h, d)];
		d_dens[flattenIdx(col, row-1, stk, w, h, d)] = d_dens[flattenIdx(col, row-2, stk, w, h, d)];
		d_dens[flattenIdx(col, row, stk-1, w, h, d)] = d_dens[flattenIdx(col, row, stk-2, w, h, d)];
		//printf("%f\n", d_dens[flattenIdx(0, row, stk, w, h, d)]);
		// momPx case: first and last cells are equal to the second and last-1 cells
		d_momPx[flattenIdx(0, row, stk, w, h, d)] = d_momPx[flattenIdx(1, row, stk, w, h, d)];
		d_momPx[flattenIdx(col, 0, stk, w, h, d)] = d_momPx[flattenIdx(col, 1, stk, w, h, d)];
		d_momPx[flattenIdx(col, row, 0, w, h, d)] = d_momPx[flattenIdx(col, row, 1, w, h, d)];
		d_momPx[flattenIdx(col-1, row, stk, w, h, d)] = d_momPx[flattenIdx(col-2, row, stk, w, h, d)];
		d_momPx[flattenIdx(col, row-1, stk, w, h, d)] = d_momPx[flattenIdx(col, row-2, stk, w, h, d)];
		d_momPx[flattenIdx(col, row, stk-1, w, h, d)] = d_momPx[flattenIdx(col, row, stk-2, w, h, d)];
		// momPy case: first and last cells are equal to the second and last-1 cells
		d_momPy[flattenIdx(0, row, stk, w, h, d)] = d_momPy[flattenIdx(1, row, stk, w, h, d)];
		d_momPy[flattenIdx(col, 0, stk, w, h, d)] = d_momPy[flattenIdx(col, 1, stk, w, h, d)];
		d_momPy[flattenIdx(col, row, 0, w, h, d)] = d_momPy[flattenIdx(col, row, 1, w, h, d)];
		d_momPy[flattenIdx(col-1, row, stk, w, h, d)] = d_momPy[flattenIdx(col-2, row, stk, w, h, d)];
		d_momPy[flattenIdx(col, row-1, stk, w, h, d)] = d_momPy[flattenIdx(col, row-2, stk, w, h, d)];
		d_momPy[flattenIdx(col, row, stk-1, w, h, d)] = d_momPy[flattenIdx(col, row, stk-2, w, h, d)];
		// momPz case: first and last cells are equal to the second and last-1 cells
		d_momPz[flattenIdx(0, row, stk, w, h, d)] = d_momPz[flattenIdx(1, row, stk, w, h, d)];
		d_momPz[flattenIdx(col, 0, stk, w, h, d)] = d_momPz[flattenIdx(col, 1, stk, w, h, d)];
		d_momPz[flattenIdx(col, row, 0, w, h, d)] = d_momPz[flattenIdx(col, row, 1, w, h, d)];
		d_momPz[flattenIdx(col-1, row, stk, w, h, d)] = d_momPz[flattenIdx(col-2, row, stk, w, h, d)];
		d_momPz[flattenIdx(col, row-1, stk, w, h, d)] = d_momPz[flattenIdx(col, row-2, stk, w, h, d)];
		d_momPz[flattenIdx(col, row, stk-1, w, h, d)] = d_momPz[flattenIdx(col, row, stk-2, w, h, d)];
		// Bx case: first and last cells are equal to the second and last-1 cells
		d_Bx[flattenIdx(0, row, stk, w, h, d)] = d_Bx[flattenIdx(1, row, stk, w, h, d)];
		d_Bx[flattenIdx(col, 0, stk, w, h, d)] = d_Bx[flattenIdx(col, 1, stk, w, h, d)];
		d_Bx[flattenIdx(col, row, 0, w, h, d)] = d_Bx[flattenIdx(col, row, 1, w, h, d)];
		d_Bx[flattenIdx(col-1, row, stk, w, h, d)] = d_Bx[flattenIdx(col-2, row, stk, w, h, d)];
		d_Bx[flattenIdx(col, row-1, stk, w, h, d)] = d_Bx[flattenIdx(col, row-2, stk, w, h, d)];
		d_Bx[flattenIdx(col, row, stk-1, w, h, d)] = d_Bx[flattenIdx(col, row, stk-2, w, h, d)];
		// By case: first and last cells are equal to the second and last-1 cells
		d_By[flattenIdx(0, row, stk, w, h, d)] = d_By[flattenIdx(1, row, stk, w, h, d)];
		d_By[flattenIdx(col, 0, stk, w, h, d)] = d_By[flattenIdx(col, 1, stk, w, h, d)];
		d_By[flattenIdx(col, row, 0, w, h, d)] = d_By[flattenIdx(col, row, 1, w, h, d)];
		d_By[flattenIdx(col-1, row, stk, w, h, d)] = d_By[flattenIdx(col-2, row, stk, w, h, d)];
		d_By[flattenIdx(col, row-1, stk, w, h, d)] = d_By[flattenIdx(col, row-2, stk, w, h, d)];
		d_By[flattenIdx(col, row, stk-1, w, h, d)] = d_By[flattenIdx(col, row, stk-2, w, h, d)];
		// Bz case: first and last cells are equal to the second and last-1 cells
		d_Bz[flattenIdx(0, row, stk, w, h, d)] = d_Bz[flattenIdx(1, row, stk, w, h, d)];
		d_Bz[flattenIdx(col, 0, stk, w, h, d)] = d_Bz[flattenIdx(col, 1, stk, w, h, d)];
		d_Bz[flattenIdx(col, row, 0, w, h, d)] = d_Bz[flattenIdx(col, row, 1, w, h, d)];
		d_Bz[flattenIdx(col-1, row, stk, w, h, d)] = d_Bz[flattenIdx(col-2, row, stk, w, h, d)];
		d_Bz[flattenIdx(col, row-1, stk, w, h, d)] = d_Bz[flattenIdx(col, row-2, stk, w, h, d)];
		d_Bz[flattenIdx(col, row, stk-1, w, h, d)] = d_Bz[flattenIdx(col, row, stk-2, w, h, d)];
		// e case: first and last cells are equal to the second and last-1 cells
		d_e[flattenIdx(0, row, stk, w, h, d)] = d_e[flattenIdx(1, row, stk, w, h, d)];
		d_e[flattenIdx(col, 0, stk, w, h, d)] = d_e[flattenIdx(col, 1, stk, w, h, d)];
		d_e[flattenIdx(col, row, 0, w, h, d)] = d_e[flattenIdx(col, row, 1, w, h, d)];
		d_e[flattenIdx(col-1, row, stk, w, h, d)] = d_e[flattenIdx(col-2, row, stk, w, h, d)];
		d_e[flattenIdx(col, row-1, stk, w, h, d)] = d_e[flattenIdx(col, row-2, stk, w, h, d)];
		d_e[flattenIdx(col, row, stk-1, w, h, d)] = d_e[flattenIdx(col, row, stk-2, w, h, d)];
		__syncthreads();
		return;
	}
*/
	// Return values at the comet position
	if (dist <= Rn){
		return;
	}

	// Normalization of physical quantities
	float distN=dist/lambda;
	float dtN = tNorm(dt);//Time normalization
//	if (d_dens[idx] < densNorm(rhowind)) d_dens[idx] = densNorm(rhowind);

	double densN = d_dens[idx];//previously normalized
	// + cosf(IMFang),sinf(IMFang)
	Vec3d vN = {d_momPx[idx]/d_dens[idx], d_momPy[idx]/d_dens[idx], d_momPz[idx]/d_dens[idx]};
//	if (col==0){
	//	vN ={10.f, 0.f, 0.f};	};
	Vec3d vn = {xyz.x/dist, xyz.y/dist, xyz.z/dist};	// terminal velocity of neutral gas [km/s]
	Vec3d vnN = vNorm(vn);
	float temp = eTemp(distN);
//	float pN = 0.5*(d_dens[idx]*dotProd(vN,vN)/3.f);
	float pN = (d_dens[idx]*rhowind)*Boltz*eTemp(dist)/(mc*amu2kg)/
			(rhowind*awind*awind*powf(mPerkm,2));
//	float pN = (gamma-1.f)*d_dens[idx]*(0.5f*(densN*dotProd(vN,vN) + 2.f*pN/(gamma-1.f) + dotProd(BN,BN)))
	//add the other pressure form
	Vec3d BN = {d_Bx[idx],// + IMF*1.e-9f/sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2)),
			d_By[idx],// + IMF*1.e-9f/sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2)),
			d_Bz[idx]};
	float aN = sqrtf(gamma*pN/densN);
//	printf("%f\n",distN);
//	if (dist <= 20000.f) {printf("%f\t%d\n", dist, idx);}
/*	if (idx==1056704) {
//		printf("%f\n",dist);
//		printf("%f\t%f\t%f\n", d_dens[idx], 255.f*(1.f - d_dens[idx]), 255.f*d_dens[idx]);
		double m = mc*amu2kg;//d_dens[idx]*rhowind;
		double v = sqrtf(d_momPx[idx]*d_momPx[idx] +
				d_momPy[idx]*d_momPy[idx] +
				d_momPz[idx]*d_momPz[idx])/d_dens[idx]*(awind*mPerkm);
		double q=charge;
//		double B=d_Bx[idx]*sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2));
//		double gyroR = (m*v)/(q*B)/mPerkm;
//		double q = charge*mc;
//		double B = sqrtf(powf(d_Bx[idx],2.f) + powf(d_By[idx],2.f) +
//    			powf(d_Bz[idx],2.f))*(sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2)));
//		printf("%.28f\t%f\t%.20f\t%.15f\t%f\n", m, v, q, B, (m*v)/(q*B)/mPerkm);
//		printf("%f\n", B/1.e-9f);
//		printf("%f\t%f\t%f\n", d_e[idx], 255.f*(1.f - d_e[idx]), 255.f*d_e[idx]);
//		printf("%f\t%f\t%f\n", gyroR,
//				255.f*(1.f - __log10f(gyroR)/4.f),
//				255.f*__log10f(gyroR)/4.f);
//		printf("%f\t%f\n",dist,temp);
//		printf("%f\t%f\t%f\n", vN.x*awind,
//				255.f*(1.f - vN.x/1.f),
//				255.f*vN.x/1.f);
//		double vp=v;
//		double densp=d_dens[idx]*rhowind;///(mc*amu2kg);
//		double densn = Q * expf(-dist / lambda)/(4.f * PI * vel_n * dist * dist * (powf(mPerkm,3)));
//		double dObs = 1.e+9f;	// observation distance[m]
//		printf("%f\n",dist);
//		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\n", vN.x, vN.y, vN.z,
//				sqrtf(vN.x*vN.x + vN.y*vN.y +vN.z*vN.z),
//				xyz.x, xyz.y, xyz.z);
//		printf("%f\t%.f\t%.20f\t%f\t%f\n", dist, densn, densp, vp, densn*densp*vp*qecsH*dObs);
//		printf("%f\t%f\t%f\n", __log10f(mc*d_momPx[idx]*d_dens[idx]*awind),
//				255.f*(1.f - __log10f(mc*d_momPx[idx]*d_dens[idx]*awind)/5.f),
//				255.f*__log10f(mc*d_momPx[idx]*d_dens[idx]*awind)/5.f);
//		double  Bx=d_Bx[idx]*sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2))/1.e-9f;
//		printf("%d\t%f\t%f\t%f\n", time, Bx,
//				255.f*(1.f - Bx/IMF),
//				255.f*(Bx/IMF));
//		printf("%f\t%f\t%f\t%f\n\n", d_Bx[idx+1]*sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2))/1.e-9f,
//				255.f*(1.f - __log10f(d_Bx[idx+1]*sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2))/1.e-9f)/5.f),
//				255.f*__log10f(d_Bx[idx+1]*sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2))/1.e-9f)/5.f);
//		printf("%f\n",pN);
		//(rhowind*awind*awind*powf(mPerkm,2))/rhowind
//		printf("%f\t%f\t%f\n", d_e[idx],//*(rhowind*awind*awind*powf(mPerkm,2))/rhowind,
//				255.f*(1.f - d_e[idx]),
//				255.f*d_e[idx]);
	}
*/
//	printf("%f\t%f\t%f\n", d_dens[idx], 255.f*(1.f - d_dens[idx]), 255.f*d_dens[idx]);
//	if (dist <= 500.f) printf("%f\n", dist);


	OctVec src = Source( densN, vN, vnN,  BN,  pN, distN);
	OctVec loss = Loss( densN, vN, pN, distN);
//	OctVec state = State( densN, vN, BN, pN);
//	if (idx==1056704) {printf("%f\n\n", vN.x*awind);}
//	if ((xyz.x < 0.f) && (xyz.y == 0.f) && (xyz.z == 0.f)){
//		printf("%f\t%f\n", xyz.x, vN.x*awind);
//	}
/*	if ((xyz.x == 500000.f) && (xyz.y == 0.f) && (xyz.z == 0.f)){
		printf("%f\t%f\n", xyz.x, vN.x*awind);
	}
*/
	//d_dens[flattenIdx(col, row, stk, w, h, d)] = densN;

	OctVec stateL = {
			d_dens[idx-1],
			d_momPx[idx-1],
			d_momPy[idx-1],
			d_momPz[idx-1],
			d_Bx[idx-1],
			d_By[idx-1],
			d_Bz[idx-1],
			d_e[idx-1]};
	__syncthreads();

	OctVec stateR = {
			d_dens[idx],
			d_momPx[idx],
			d_momPy[idx],
			d_momPz[idx],
			d_Bx[idx],
			d_By[idx],
			d_Bz[idx],
			d_e[idx]};
	__syncthreads();

	OctVec fluxLeft = FluxLeft(densN, vN, BN, stateL);
	OctVec fluxRight = FluxRight(densN, vN, BN, stateR);

	//Call to Right EigenVectors
	OctVec EWE = EWEVec(vN);
	OctVec plusAWE = plusAWEVec(densN, vN, BN);
	OctVec minusAWE = minusAWEVec(densN, vN, BN);
	OctVec plusSlowMAE = plusSlowMAEVec(densN, vN, BN, aN);
	OctVec minusSlowMAE = minusSlowMAEVec(densN, vN, BN, aN);
	OctVec plusFastMAE = plusFastMAEVec(densN, vN, BN, aN);
	OctVec minusFastMAE = minusFastMAEVec(densN, vN, BN, aN);
	OctVec DMFE = DMFEVec(BN);

	//Call to Eigenvalues
	float EigValEWE = EWEVal(vN);
	float EigValplusAWE = plusAWEVal(densN, vN, BN);
	float EigValminusAWE = minusAWEVal(densN, vN, BN);
	float EigValplusSlowMAE = plusSlowMAEVal(densN, vN, BN, aN);
	float EigValminusSlowMAE = minusSlowMAEVal(densN, vN, BN, aN);
	float EigValplusFastMAE = plusFastMAEVal(densN, vN, BN, aN);
	float EigValminusFastMAE = minusFastMAEVal(densN, vN, BN, aN);
	float EigValDMFE = DMFEVal(vN);

/*	OctVec stateDiff = {
			d_dens[idx-1] - d_dens[idx],
			d_momPx[idx-1] - d_momPx[idx],
			d_momPy[idx-1] - d_momPy[idx],
			d_momPz[idx-1] - d_momPz[idx],
			d_Bx[idx-1] - d_Bx[idx],
			d_By[idx-1] - d_By[idx],
			d_Bz[idx-1] - d_Bz[idx],
			d_e[idx-1] - d_e[idx]
	};
*/
	OctVec eigVleftDotStateDiff = {
		(d_dens[idx-1] - d_dens[idx])*(EWE.dens +
				plusAWE.dens +
				minusAWE.dens +
				plusSlowMAE.dens +
				minusSlowMAE.dens +
				plusFastMAE.dens +
				minusFastMAE.dens +
				DMFE.dens),
		(d_momPx[idx-1] - d_momPx[idx])*(EWE.momPx +
				plusAWE.momPx +
				minusAWE.momPx +
				plusSlowMAE.momPx +
				minusSlowMAE.momPx +
				plusFastMAE.momPx +
				minusFastMAE.momPx +
				DMFE.momPx),
		(d_momPy[idx-1] - d_momPy[idx])*(EWE.momPy +
				plusAWE.momPy +
				minusAWE.momPy +
				plusSlowMAE.momPy +
				minusSlowMAE.momPy +
				plusFastMAE.momPy +
				minusFastMAE.momPy +
				DMFE.momPy),
		(d_momPz[idx-1] - d_momPz[idx])*(EWE.momPz +
				plusAWE.momPz +
				minusAWE.momPz +
				plusSlowMAE.momPz +
				minusSlowMAE.momPz +
				plusFastMAE.momPz +
				minusFastMAE.momPz +
				DMFE.momPz),
		(d_Bx[idx-1] - d_Bx[idx])*(EWE.Bx +
				plusAWE.Bx +
				minusAWE.Bx +
				plusSlowMAE.Bx +
				minusSlowMAE.Bx +
				plusFastMAE.Bx +
				minusFastMAE.Bx +
				DMFE.Bx),
		(d_By[idx-1] - d_By[idx])*(EWE.By +
				plusAWE.By +
				minusAWE.By +
				plusSlowMAE.By +
				minusSlowMAE.By +
				plusFastMAE.By +
				minusFastMAE.By +
				DMFE.By),
		(d_Bz[idx-1] - d_Bz[idx])*(EWE.Bz +
				plusAWE.Bz +
				minusAWE.Bz +
				plusSlowMAE.Bz +
				minusSlowMAE.Bz +
				plusFastMAE.Bz +
				minusFastMAE.Bz +
				DMFE.momPz),
		(d_e[idx-1] - d_e[idx])*(EWE.e +
				plusAWE.e +
				minusAWE.e +
				plusSlowMAE.e +
				minusSlowMAE.e +
				plusFastMAE.e +
				minusFastMAE.e +
				DMFE.e)};

	//if (idx==1000000)	printf("%.35f\n", eigVleftDotStateDiff.dens);

	OctVec meanFluxCell = {
			0.5f*(fluxLeft.dens + fluxRight.dens),
			0.5f*(fluxLeft.momPx + fluxRight.momPx),
			0.5f*(fluxLeft.momPy + fluxRight.momPy),
			0.5f*(fluxLeft.momPz + fluxRight.momPz),
			0.5f*(fluxLeft.Bx + fluxRight.Bx),
			0.5f*(fluxLeft.By + fluxRight.By),
			0.5f*(fluxLeft.Bz + fluxRight.Bz),
			0.5f*(fluxLeft.e + fluxRight.e)};

	//if (idx==1000000)	printf("%.35f\n", meanFluxCell.dens);

	OctVec fluxCell = {
			meanFluxCell.dens - (eigVleftDotStateDiff.dens*fabsf(EWEVal(vN))*EWE.dens +
				eigVleftDotStateDiff.dens*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.dens +
				eigVleftDotStateDiff.dens*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.dens +
				eigVleftDotStateDiff.dens*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.dens +
				eigVleftDotStateDiff.dens*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.dens +
				eigVleftDotStateDiff.dens*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.dens +
				eigVleftDotStateDiff.dens*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.dens +
				eigVleftDotStateDiff.dens*fabsf(DMFEVal(vN))*DMFE.dens),
			meanFluxCell.momPx - (eigVleftDotStateDiff.momPx*fabsf(EWEVal(vN))*EWE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.momPx +
				eigVleftDotStateDiff.momPx*fabsf(DMFEVal(vN))*DMFE.momPx),
			meanFluxCell.momPy - (eigVleftDotStateDiff.momPy*fabsf(EWEVal(vN))*EWE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.momPy +
				eigVleftDotStateDiff.momPy*fabsf(DMFEVal(vN))*DMFE.momPy),
			meanFluxCell.momPz - (eigVleftDotStateDiff.momPz*fabsf(EWEVal(vN))*EWE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.momPz +
				eigVleftDotStateDiff.momPz*fabsf(DMFEVal(vN))*DMFE.momPz),
			meanFluxCell.Bx - (eigVleftDotStateDiff.Bx*fabsf(EWEVal(vN))*EWE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.Bx +
				eigVleftDotStateDiff.Bx*fabsf(DMFEVal(vN))*DMFE.Bx),
			meanFluxCell.By - (eigVleftDotStateDiff.By*fabsf(EWEVal(vN))*EWE.By +
				eigVleftDotStateDiff.By*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.By +
				eigVleftDotStateDiff.By*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.By +
				eigVleftDotStateDiff.By*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.By +
				eigVleftDotStateDiff.By*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.By +
				eigVleftDotStateDiff.By*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.By +
				eigVleftDotStateDiff.By*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.By +
				eigVleftDotStateDiff.By*fabsf(DMFEVal(vN))*DMFE.By),
			meanFluxCell.Bz - (eigVleftDotStateDiff.Bz*fabsf(EWEVal(vN))*EWE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.Bz +
				eigVleftDotStateDiff.Bz*fabsf(DMFEVal(vN))*DMFE.Bz),
			meanFluxCell.e - (eigVleftDotStateDiff.e*fabsf(EWEVal(vN))*EWE.e +
				eigVleftDotStateDiff.e*fabsf(plusAWEVal(densN, vN, BN))*plusAWE.e +
				eigVleftDotStateDiff.e*fabsf(minusAWEVal(densN, vN, BN))*minusAWE.e +
				eigVleftDotStateDiff.e*fabsf(plusSlowMAEVal(densN, vN, BN, aN))*plusSlowMAE.e +
				eigVleftDotStateDiff.e*fabsf(minusSlowMAEVal(densN, vN, BN, aN))*minusSlowMAE.e +
				eigVleftDotStateDiff.e*fabsf(plusFastMAEVal(densN, vN, BN, aN))*plusFastMAE.e +
				eigVleftDotStateDiff.e*fabsf(minusFastMAEVal(densN, vN, BN, aN))*minusFastMAE.e +
				eigVleftDotStateDiff.e*fabsf(DMFEVal(vN))*DMFE.e)};

	//if (idx==1000000)	printf("%.35f\n", fluxCell.dens);
	//if (idx==1000000)	printf("%.35f\t", d_dens[idx]);

		//copy the new physical quantities to arrays
	d_dens[idx] = d_dens[idx] +
			(-(1.f/volCell)*(fluxCell.dens)*areaFluxX + src.dens - loss.dens)*dtN;
	//if (d_dens[idx] <= 0.f) d_dens[idx] = 1.f;
	d_momPx[idx] = d_momPx[idx] +
			(-(1.f/volCell)*(fluxCell.momPx)*areaFluxX + src.momPx - loss.momPx)*dtN;
	d_momPy[idx] = d_momPy[idx] +
			(-(1.f/volCell)*(fluxCell.momPy)*areaFluxY + src.momPy - loss.momPy)*dtN;
	d_momPz[idx] = d_momPz[idx] +
			(-(1.f/volCell)*(fluxCell.momPz)*areaFluxZ + src.momPz - loss.momPz)*dtN;
	d_Bx[idx] = d_Bx[idx] +
			(-(1.f/volCell)*(fluxCell.Bx)*areaFluxX + src.Bx - loss.Bx)*dtN;
	d_By[idx] = d_By[idx] +
			(-(1.f/volCell)*(fluxCell.By)*areaFluxY + src.By - loss.By)*dtN;
	d_Bz[idx] = d_Bz[idx] +
			(-(1.f/volCell)*(fluxCell.Bz)*areaFluxZ + src.Bz - loss.Bz)*dtN;
	d_e[idx] = d_e[idx] +
			(-(1.f/volCell)*(fluxCell.e)*areaFluxX + src.e - loss.e)*dtN;

	//if (idx == 1000000)	printf("%.35f\t%.35f\t%.35f\n", d_dens[idx], (1.f/volCell)*(fluxCell.dens)*areaFluxX*dtN, (src.dens - loss.dens) * dtN);

//	if (d_dens[idx] < 0.f) d_dens[idx] =1,f
//	if (d_Bx[idx] < 0.f) d_Bx[idx] = 1.f;
//	if (d_By[idx] < 0.f) d_By[idx] = 1.f;
//	if (d_Bz[idx] < 0.f) d_Bz[idx] = 1.f;
//	if (d_e[idx] < 0.f) d_e[idx] = 1.f;

	//if (idx==1000000)	printf("%.35f\t%.35f\t%.35f\t%.35f\t%.35f\t%.35f\n",(gamma - 1.f)*d_dens[idx]*d_e[idx] ,(1.f/volCell)*(fluxCell.dens)*areaFluxX*lambda, src.dens, loss.dens, fluxLeft.dens, meanFluxCell.dens);

	//condition to hierarchical based octree structure
/*	float grad =
	if (() | () | ()){
		printf("%f\n",d_dens[idx]);
	}
*/

	__syncthreads();


}


/////////////////////////////////////////////////////////////////
//  L A U N C H E R S
void kernelLauncher(uchar4 *d_out, double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz,	double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, float time, int w, int h,
		int3 volSize, int method, float zs, float theta,
		float threshold, float dis, int id) {
	dim3 blockSize(TX_2D, TY_2D);
	dim3 gridSize(divUp(w, TX_2D), divUp(h, TY_2D));
	renderKernel<<<gridSize, blockSize>>>(d_out, d_dens, d_momPx,
			d_momPy, d_momPz, d_Bx, d_By, d_Bz, d_e, time, w, h,
			volSize, method, zs, theta, threshold, dis, id);
}

void resetKernelLauncher(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz, double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int3 volSize, int id) {
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY),
			divUp(volSize.z, TZ));
	resetKernel<<<gridSize, blockSize>>>(d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz, d_e,
			volSize, id);
}


void writeData(int output, double *h_dens, double *h_momPx,
		double *h_momPy, double *h_momPz, double *h_Bx, double *h_By,
		double *h_Bz, double *h_e, int3 volSize) {
	FILE *fp;
	char name[80];
	const int vol = volSize.x*volSize.y*volSize.z;

	// write data dens
	sprintf (name, "out1/dens%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_dens, sizeof(double), vol, fp);

	// write data momPx
	sprintf (name, "out1/momPx%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_momPx, sizeof(double), vol, fp);

	// write data momPy
	sprintf (name, "out1/momPy%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_momPy, sizeof(double), vol, fp);

	// write data momPz
	sprintf (name, "out1/momPz%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_momPz, sizeof(double), vol, fp);

	// write data Bx
	sprintf (name, "out1/Bx%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_Bx, sizeof(double), vol, fp);

	// write data By
	sprintf (name, "out1/By%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_By, sizeof(double), vol, fp);

	// write data Bz
	sprintf (name, "out1/Bz%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_Bz, sizeof(double), vol, fp);

	// write data e
	sprintf (name, "out1/e%d.dat", output);
	fp = fopen(name, "w");
	fwrite (h_e, sizeof(double), vol, fp);

	fclose(fp);
}

void evolutiveKernelLauncher(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz, double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int time, int3 volSize, int id) {
	dim3 blockSize(TX, TY, TZ);
	dim3 gridSize(divUp(volSize.x, TX), divUp(volSize.y, TY),
			divUp(volSize.z, TZ));
//	const size_t smSz = (TX + 2 * RAD)*(TY + 2 * RAD)*(TZ + 2 * RAD)*sizeof(double);
	evolutiveKernel<<<gridSize, blockSize>>>(d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz, d_e,
			time, volSize, id);

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
	    printf("CUDA error: %s\n", cudaGetErrorString(error));
	    exit(-1);
	}

	// begin the statement to print the data outputs
	int output;
	double param, fractpart, intpart;
	param = float(time)/timestep;
	fractpart = modf (param , &intpart);
	if (fractpart == 0.f){
		output = int(param);

		// memory size
		size_t size = NX*NY*NZ * sizeof(double);

		// memory allocation in host
		double* h_dens = (double*)malloc(size);
		double* h_momPx = (double*)malloc(size);
		double* h_momPy = (double*)malloc(size);
		double* h_momPz = (double*)malloc(size);
		double* h_Bx = (double*)malloc(size);
		double* h_By = (double*)malloc(size);
		double* h_Bz = (double*)malloc(size);
		double* h_e = (double*)malloc(size);

		// copy data from device to host
		cudaMemcpy(h_dens, d_dens, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_momPx, d_momPx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_momPy, d_momPy, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_momPz, d_momPz, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Bx, d_Bx, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_By, d_By, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Bz, d_Bz, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_e, d_e, size, cudaMemcpyDeviceToHost);

		// call funtion to print
		//writeData(output, h_dens, h_momPx, h_momPy, h_momPz,
		//		h_Bx, h_By,	h_Bz, h_e, volSize);

		//printf("%.35f\t%.35f\n", float(h_dens[1000000]), float(h_dens[1000005]));


		// free the allocated memory
		free(h_dens);
		free(h_momPx);
		free(h_momPy);
		free(h_momPz);
		free(h_Bx);
		free(h_By);
		free(h_Bz);
		free(h_e);

	}

}

