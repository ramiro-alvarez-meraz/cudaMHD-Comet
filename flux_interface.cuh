/*
 * flux_interface.cuh
 *
 *  Created on: Oct 6, 2019
 *      Author: ramiro
 */

#ifndef FLUX_INTERFACE_CUH_
#define FLUX_INTERFACE_CUH_

__device__ float vAlfx(float dens, Vec3d B);
__device__ float cubeSound(float dens, Vec3d B, float a);
__device__ float cubeSoundSx(float dens, Vec3d B, float a);
__device__ float cubeSoundFx(float dens, Vec3d B, float a);
__device__ float alphaS(float dens, Vec3d B, float a);
__device__ float alphaF(float dens, Vec3d B, float a);
__device__ float betaY(Vec3d B);
__device__ float betaZ(Vec3d B);
__device__ float sign(float x);
__device__ float dotProd(Vec3d v, Vec3d w);
__device__ OctVec EWEVec(Vec3d v);
__device__ OctVec plusAWEVec(float dens, Vec3d v, Vec3d B);
__device__ OctVec minusAWEVec(float dens, Vec3d v, Vec3d B);
__device__ OctVec plusSlowMAEVec(float dens, Vec3d v, Vec3d B, float a);
__device__ OctVec minusSlowMAEVec(float dens, Vec3d v, Vec3d B, float a);
__device__ OctVec plusFastMAEVec(float dens, Vec3d v, Vec3d B, float a);
__device__ OctVec minusFastMAEVec(float dens, Vec3d v, Vec3d B, float a);
__device__ OctVec DMFEVec(Vec3d B);
__device__ float EWEVal(Vec3d V);
__device__ float plusAWEVal(float dens, Vec3d v, Vec3d B);
__device__ float minusAWEVal(float dens, Vec3d v, Vec3d B);
__device__ float plusSlowMAEVal(float dens, Vec3d v, Vec3d B, float a);
__device__ float minusSlowMAEVal(float dens, Vec3d v, Vec3d B, float a);
__device__ float plusFastMAEVal(float dens, Vec3d v, Vec3d B, float a);
__device__ float minusFastMAEVal(float dens, Vec3d v, Vec3d B, float a);
__device__ float DMFEVal(Vec3d v);
__device__ OctVec Mult(float dens, Vec3d v, Vec3d B, float a, OctVec fluxLeft, OctVec fluxRight);
__device__ OctVec State(float dens, Vec3d v, Vec3d B, float p);
__device__ OctVec Source(double dens, Vec3d v, Vec3d vn, Vec3d B, double p, double dist);
__device__ double eTemp(double dist);
__device__ OctVec Loss(double dens, Vec3d v, double p, double dist);

__device__ OctVec stateDiff(OctVec fluxLeft, OctVec fluxRight);
__device__ OctVec FluxLeft(double dens, Vec3d v, Vec3d B, OctVec stateLeft);
__device__ OctVec FluxRight(double dens, Vec3d v, Vec3d B, OctVec stateRight);
__device__ OctVec LeftEigenVec(float dens, Vec3d v, Vec3d B, float a);
__device__ OctVec multEigenVec(float dens, Vec3d v, Vec3d B, float a);
//__device__ OctVec leftEig();



#endif /* FLUX_INTERFACE_CUH_ */
