/*
 * device_funcs.cuh
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#ifndef FUNCS_CUH_
#define FUNCS_CUH_

typedef struct {
float3 o, d; // origin and direction
} Ray;


__host__ int divUp(int a, int b);
__device__ unsigned char clip(int n);
__device__ float3 yRotate(float3 pos, float theta);
__device__ int flatten(int3 index, int3 volSize);
__device__ int3 posToVolIndex(double3 pos, int3 volSize);
__device__ int flattenIdx(int col, int row, int stk, int width, int height, int hight);
__device__ float neutralDens(float dist);
__device__ float electronDens(float dist);
__device__ float ionDens(float dist);
__device__ float density(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz,	double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int3 volSize, float3 pos, int id);
__device__ float3 paramRay(Ray r, float t);
__device__ float3 scrIdxToPos(int c, int r, int w, int h, float zs);
__device__ bool rayPlaneIntersect(Ray myRay, float3 n, float dist, float *t);
__device__ bool intersectBox(Ray r, float3 boxmin, float3 boxmax,
		float *tnear, float *tfar);
__device__ uchar4 sliceShader(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz,	double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int3 volSize, Ray boxRay,
		float threshold, float dist, float3 norm, int id);
//__device__ uchar4 volumeRenderShader(float *d_dens, Vec3d *d_momP, Vec3d *d_B, float *d_e, int3 volSize,
//		Ray boxRay, float dist, int numSteps, int id);
//__device__ uchar4 rayCastShader(float *d_dens, Vec3d *d_momP, Vec3d *d_B, float *d_e, int3 volSize,
//		Ray boxRay, float dist, int id);
__device__ float scale(int i, int w);
__device__ float tNorm(float t);
__device__ Vec3d coordNorm(Vec3d X);
__device__ float dNorm(Vec3d X);
__device__ float densNorm(float dens);
__device__ Vec3d vNorm(Vec3d vel);
__device__ float pNorm(float X);
__device__ Vec3d BNorm(Vec3d B);
__device__ float dotProd(Vec3d V, Vec3d W);



#endif /* FUNCS_CUH_ */
