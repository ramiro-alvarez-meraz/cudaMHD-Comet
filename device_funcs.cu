/*
 * device_funcs.cu
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#include "struct.h"
#include "device_funcs.cuh"
#include "helper_math.h"
#include <stdio.h>
#include "const.h"
#include "params.h"


__host__ int divUp(int a, int b) { return (a + b - 1)/b; }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__ int clipWithBounds(int n, int n_min, int n_max) {
  return n > n_max ? n_max : (n < n_min ? n_min : n);
}

__device__ float3 yRotate(float3 pos, float theta) {
  const float c = cosf(theta), s = sinf(theta);
  return make_float3(c*pos.x + s*pos.z, pos.y, -s*pos.x + c*pos.z);
}

__device__ float3 scrIdxToPos(int c, int r, int w, int h, float zs) {
  return make_float3(c - w / 2, r - h / 2, zs);
}

__device__ float3 paramRay(Ray r, float t) { return r.o + t*(r.d); }

__device__ float planeSDF(float3 pos, float3 norm, float d) {
  return dot(pos, normalize(norm)) - d;
}

__device__
bool rayPlaneIntersect(Ray myRay, float3 n, float dist, float *t) {
  const float f0 = planeSDF(paramRay(myRay, 0.f), n, dist);
  const float f1 = planeSDF(paramRay(myRay, 1.f), n, dist);
  bool result = (f0*f1 < 0);
  if (result) *t = (0.f - f0) / (f1 - f0);
  return result;
}

// Intersect ray with a box from volumeRender SDK sample.
__device__ bool intersectBox(Ray r, float3 boxmin, float3 boxmax,
  float *tnear, float *tfar) {
  // Compute intersection of ray with all six bbox planes.
  const float3 invR = make_float3(1.0f) / r.d;
  const float3 tbot = invR*(boxmin - r.o), ttop = invR*(boxmax - r.o);
  // Re-order intersections to find smallest and largest on each axis.
  const float3 tmin = fminf(ttop, tbot), tmax = fmaxf(ttop, tbot);
  // Find the largest tmin and the smallest tmax.
  *tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
  *tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
  return *tfar > *tnear;
}

__device__ int3 posToVolIndex(float3 pos, int3 volSize) {
  return make_int3(pos.x + volSize.x/2, pos.y + volSize.y/2,
                   pos.z + volSize.z/2);
}

__device__ int flatten(int3 index, int3 volSize) {
  return index.x + index.y*volSize.x + index.z*volSize.x*volSize.y;
}

__device__
int idxClip(int idx, int idxMax) {
  return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flattenIdx(int col, int row, int stk, int width, int height, int hight) {
  return idxClip(col, width) + idxClip(row, height)*width + idxClip(stk, hight)*width*height;
}

__device__ float energyDens(){
	float ekin = 0.f;
	float etherm = 0.f;
	float emag = 0.f;
	return (ekin + etherm + emag)/2.f;
}

__device__ float electronDens(float dist){
	const float dividend = fi * Q * expf(-dist / lambda) * dt;
	const float divisor = 4.f * PI * lambda * dist * dist;
	return dividend / divisor;
}

__device__ float neutralDens(float dist){
	float dividend = (mc*amu2kg) * Q * expf(-dist / lambda);
	float divisor = 4.f * PI * vel_n * dist * dist * (powf(mPerkm,3));
	return dividend / divisor;
}

__device__ float ionDens(float dist){
	float dividend = (mc * amu2kg) * Q * fi * expf(-dist / lambda)*dt;
	float divisor = 4.f * PI * lambda * dist * dist * pow(mPerkm, 3);
	return dividend / divisor;
}

__device__ float density(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz, double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int3 volSize, float3 pos, int id) {
	  int3 index = posToVolIndex(pos, volSize);
	  int i = index.x, j = index.y, k = index.z;
	  const int w = volSize.x, h = volSize.y, d = volSize.z;
	  //const float3 rem = fracf(pos);
	  index = make_int3(clipWithBounds(i, 0, w - 2),
	    clipWithBounds(j, 0, h - 2), clipWithBounds(k, 0, d - 2));
//	const int3 pos0 = { w/2, h/2, d/2 };
//	const float dx = index.x - pos0.x, dy = index.y - pos0.y, dz = index.z - pos0.z;
//	const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
	double quantity = 0.f;
	double m = 0.f;
	double v = 0.f;
	double q = 0.f;
	double B = 0.f;
//	d_dens[i] = ionDens(dist);
    if (id == 0){	// mass density
    	quantity=d_dens[flatten(index, volSize)];}
    if (id == 1){	// (mean) lineal momentum
    	quantity=mc*sqrtf(powf(d_momPx[flatten(index, volSize)],2.f) +
    			powf(d_momPy[flatten(index, volSize)],2.f) +
    			powf(d_momPz[flatten(index, volSize)],2.f))/d_dens[flatten(index, volSize)]*awind;}
    if (id == 2){	// (mean) magnetic field
    	quantity=sqrtf(powf(d_Bx[flatten(index, volSize)],2.f) +
    			powf(d_By[flatten(index, volSize)],2.f) +
    			powf(d_Bz[flatten(index, volSize)],2.f))*
    			sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2))/1.e-9f;}
    if (id == 3){	// pressure density
    	quantity=(gamma - 1.f)*d_dens[flatten(index, volSize)]*d_e[flatten(index, volSize)];}//*(rhowind*awind*awind*powf(mPerkm,2))/rhowind;}
    if (id == 4){	// Gyroradius (in units of km)
    	m = mc*amu2kg;//(d_dens[flatten(index, volSize)]*rhowind)*powf(mPerkm,3)*volCell/amu2kg;
    	v = sqrtf(powf(d_momPx[flatten(index, volSize)],2.f) +
    			powf(d_momPy[flatten(index, volSize)],2.f) +
    			powf(d_momPz[flatten(index, volSize)],2.f))/
    			d_dens[flatten(index, volSize)]*awind*mPerkm;
    	q = charge*mc;	//ion charge is proportional to the mean molecular mass of cometary ions
    	B = sqrtf(powf(d_Bx[flatten(index, volSize)],2.f) +
    			powf(d_By[flatten(index, volSize)],2.f) +
    			powf(d_Bz[flatten(index, volSize)],2.f))*(sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2)));
    	quantity=(m*v)/(q*B)/mPerkm;}
    if (id == 5){	// x-velocity component
    	quantity=d_momPx[flatten(index, volSize)]/d_dens[flatten(index, volSize)]*awind;}
    if (id == 6){	// ENAs production
		double vp=sqrtf(powf(d_momPx[flatten(index, volSize)],2.f) +
    			powf(d_momPy[flatten(index, volSize)],2.f) +
    			powf(d_momPz[flatten(index, volSize)],2.f))/
    			d_dens[flatten(index, volSize)]*awind*mPerkm;
		double densp=d_dens[flatten(index, volSize)]*rhowind/(mc*amu2kg);
		Vec3d xyz = {scale(i, w), scale(j, h), scale(k, d)};// Scale thread to distance
//		double dist = (double)sqrtf(dotProd(xyz, xyz));
		double dist = sqrtf(xyz.x*xyz.x + xyz.y*xyz.y + xyz.z*xyz.z);
		double densn = Q * expf(-dist / lambda)/(4.f * PI * vel_n * dist * dist * (powf(mPerkm,3)));
		double dObs = 1.e+9f;	// observation distance[m]
    	quantity=densn*densp*vp*qecsH*dObs;}

 //   if ((id==6)) printf("%f\n", quantity);//__log10f(quantity*rhowind/(amu2kg)));
    //return quantity
    if (id == 0) return __log10f(quantity);//*rhowind/(amu2kg);
    if (id == 1) return __log10f(quantity);
    if (id == 2) return quantity;
    if (id == 3) return quantity;
    if (id == 4) return __log10f(quantity);
    if (id == 5) return __log10f(quantity);
    if (id == 6) return __log10f(quantity);
    else return __log10f(quantity);
}

__device__ float scale(int i, int w){
	return 2 * LEN*(((1.f*i)/w) - 0.5f);
}

////////////////////////////////////////////////////
//Normalization of physical quantities:

__device__ float tNorm(float t){//time
	return awind*t/lambda;
}

__device__ Vec3d coordNorm(Vec3d X){
	return {X.x/lambda, X.y/lambda, X.z/lambda};
}

__device__ float dNorm(Vec3d X){
	return sqrtf(X.x*X.x + X.y*X.y + X.z*X.z)/lambda;
}

__device__ float densNorm(float dens){
	return dens/(rhowind);
}

__device__ Vec3d vNorm(Vec3d vel){
	return {vel.x/awind, vel.y/awind, vel.z/awind};
}

__device__ float pNorm(float X){
	return X/(rhowind*awind*awind*powf(mPerkm,2));
}

__device__ Vec3d BNorm(Vec3d B){
	//B en nT
	const float nT2T = 1.e-9f;//
	const float norm = sqrtf(mu*rhowind*awind*awind*powf(mPerkm,2));
	return {B.x*nT2T/norm,
		B.y*nT2T/norm,
		B.z*nT2T/norm};
}





