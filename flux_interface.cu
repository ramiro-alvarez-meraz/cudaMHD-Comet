/*
 * flux_interface.cu
 *
 *  Created on: Sep 30, 2019
 *      Author: ramiro
 */

#include "struct.h"
#include "flux_interface.cuh"
#include "device_funcs.cuh"
#include "helper_math.h"
#include <stdio.h>
#include "const.h"
#include "params.h"

__device__ float vAlfx(float dens, Vec3d B){
	return B.x/sqrtf(dens);
}

__device__ float cubeSound(float dens, Vec3d B, float a){
	return pow(a,2) + ((powf(B.x,2) + powf(B.y,2) + powf(B.z,2)) / dens);
}

__device__ float cubeSoundSx(float dens, Vec3d B, float a){
	return 0.5f * (cubeSound(dens,B,a) -
			sqrtf(powf(cubeSound(dens,B,a),2)-(4*a*a*powf(vAlfx(dens,B),2))));
}

__device__ float cubeSoundFx(float dens, Vec3d B, float a){
	return 0.5f * (cubeSound(dens,B,a) +
			sqrtf(powf(cubeSound(dens,B,a),2)-(4*a*a*powf(vAlfx(dens,B),2))));
}

__device__ float betaY(Vec3d B){
	return B.y/sqrtf(B.y*B.y + B.z*B.z);
}

__device__ float betaZ(Vec3d B){
	return B.z/sqrtf(B.y*B.y + B.z*B.z);
}

__device__ float alphaS(float dens, Vec3d B, float a){
	const float dividend = sqrtf(cubeSoundFx(dens,B,a) - a*a);
	const float divisor = sqrtf(cubeSoundFx(dens,B,a) - cubeSoundSx(dens,B,a));
	return dividend/divisor;
}

__device__ float alphaF(float dens, Vec3d B, float a){
	const float dividend = sqrtf(dens*cubeSoundFx(dens,B,a) - B.x*B.x);
	const float divisor = sqrtf(dens*cubeSoundFx(dens,B,a) - dens*cubeSoundSx(dens,B,a));
	return dividend/divisor;
}

__device__ float dotProd(Vec3d v, Vec3d w){
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

__device__ float sign(float x){
	float t = x<0.f ? -1.f : 0.f;
	return x > 0.f ? 1.f : t;
}



////////////////////////////////////////////////////
//Eigenvectors:

__device__ OctVec EWEVec(Vec3d v){//Entropy Wave EigenVector
	return {1.f,
		v.x,
		v.y,
		v.z,
		0.f,
		0.f,
		0.f,
		0.5f*dotProd(v,v)};
}

__device__ OctVec plusAWEVec(float dens, Vec3d v, Vec3d B){//Plus Alfven Wave EigenVector
	return {0.f,
		0.f,
		-betaZ(B),
		betaY(B),
		0.f,
		betaZ(B)/sqrtf(dens),
		-betaY(B)/sqrtf(dens),
		(v.z*betaY(B) - v.y*betaZ(B))};
}

__device__ OctVec minusAWEVec(float dens, Vec3d v, Vec3d B){//Minus Alfven Wave EigenVector
	return {0.f,
		0.f,
		betaZ(B),
		-betaY(B),
		0.f,
		betaZ(B)/sqrtf(dens),
		-betaY(B)/sqrtf(dens),
		-((v.z*betaY(B))-(v.y*betaZ(B)))};
}

__device__ float plusAlphaHslow(float dens, Vec3d v, Vec3d B, float a){
	return (alphaS(dens,B,a)*(cubeSoundSx(dens,B,a)/(gamma-1.f)+
			(sqrtf(cubeSoundSx(dens,B,a)*v.x))+((gamma-2.f)/(gamma-1.f))*(cubeSoundSx(dens,B,a)-(a*a))))+
			alphaF(dens,B,a)*((betaY(B)*v.y)+(betaZ(B)*v.z))*a*sign(B.x);
}

__device__ OctVec plusSlowMAEVec(float dens, Vec3d v, Vec3d B, float a){//Plus Slow MagnetoAcoustic Wave EigenVector
	return {alphaS(dens,B,a),
		alphaS(dens,B,a)*(v.x+sqrtf(cubeSoundSx(dens,B,a))),
		(alphaS(dens,B,a)*v.y)+(alphaF(dens,B,a)*betaY(B)*a*sign(B.x)),
		(alphaS(dens,B,a)*v.z)+(alphaF(dens,B,a)*betaZ(B)*a*sign(B.x)),
		0.f,
		-(alphaF(dens,B,a)*betaY(B)*a*a)/(sqrtf(dens)*sqrtf(cubeSoundFx(dens,B,a))),
		-(alphaF(dens,B,a)*betaZ(B)*a*a)/(sqrtf(dens)*sqrtf(cubeSoundFx(dens,B,a))),
		(alphaS(dens,B,a)*0.5f*dotProd(v,v))+plusAlphaHslow(dens,v,B,a)};
}

__device__ float minusAlphaHslow(float dens, Vec3d v, Vec3d B, float a){
	return (alphaS(dens,B,a)*(cubeSoundSx(dens,B,a)/(gamma-1.f)-
			(sqrtf(cubeSoundSx(dens,B,a)*v.x))+((gamma-2.f)/(gamma-1.f))*(cubeSoundSx(dens,B,a)-(a*a))))-
			alphaF(dens,B,a)*((betaY(B)*v.y)+(betaZ(B)*v.z))*a*sign(B.x);
}

__device__ OctVec minusSlowMAEVec(float dens, Vec3d v, Vec3d B, float a){//Minus Slow MagnetoAcoustic Wave EigenVector
	return {alphaS(dens,B,a),
		alphaS(dens,B,a)*(v.x-sqrtf(cubeSoundSx(dens,B,a))),
		(alphaS(dens,B,a)*v.y)-(alphaF(dens,B,a)*betaY(B)*a*sign(B.x)),
		(alphaS(dens,B,a)*v.z)-(alphaF(dens,B,a)*betaZ(B)*a*sign(B.x)),
		0.f,
		-(alphaF(dens,B,a)*betaY(B)*a*a)/(sqrtf(dens)*sqrtf(cubeSoundFx(dens,B,a))),
		-(alphaF(dens,B,a)*betaZ(B)*a*a)/(sqrtf(dens)*sqrtf(cubeSoundFx(dens,B,a))),
		(alphaS(dens,B,a)*0.5f*dotProd(v,v))+minusAlphaHslow(dens,v,B,a)};
}

__device__ float plusAlphaHfast(float dens, Vec3d v, Vec3d B, float a){
	return (alphaS(dens,B,a)*(cubeSoundFx(dens,B,a)/(gamma-1.f)+
			(sqrtf(cubeSoundFx(dens,B,a)*v.x))+((gamma-2.f)/(gamma-1.f))*(cubeSoundFx(dens,B,a)-(a*a))))-
			alphaS(dens,B,a)*((betaY(B)*v.y)+(betaZ(B)*v.z))*vAlfx(dens,B);
}

__device__ OctVec plusFastMAEVec(float dens, Vec3d v, Vec3d B, float a){//Plus Fast MagnetoAcoustic Wave EigenVector
	return {alphaF(dens,B,a),
		alphaF(dens,B,a)*(v.x+sqrtf(cubeSoundFx(dens,B,a))),
		(alphaF(dens,B,a)*v.y)-(alphaS(dens,B,a)*betaY(B)*vAlfx(dens,B)),
		(alphaF(dens,B,a)*v.z)-(alphaS(dens,B,a)*betaZ(B)*vAlfx(dens,B)),
		0.f,
		(alphaS(dens,B,a)*sqrtf(cubeSoundFx(dens,B,a))*betaY(B))/(sqrtf(dens)),
		(alphaS(dens,B,a)*sqrtf(cubeSoundFx(dens,B,a))*betaZ(B))/(sqrtf(dens)),
		(alphaF(dens,B,a)*0.5f*dotProd(v,v))+plusAlphaHslow(dens,v,B,a)};
}

__device__ float minusAlphaHfast(float dens, Vec3d v, Vec3d B, float a){
	return (alphaS(dens,B,a)*(cubeSoundFx(dens,B,a)/(gamma-1.f)-
			(sqrtf(cubeSoundFx(dens,B,a)*v.x))+((gamma-2.f)/(gamma-1.f))*(cubeSoundFx(dens,B,a)-(a*a))))+
			alphaS(dens,B,a)*((betaY(B)*v.y)+(betaZ(B)*v.z))*vAlfx(dens,B);
}

__device__ OctVec minusFastMAEVec(float dens, Vec3d V, Vec3d B, float a){//Minus Fast MagnetoAcoustic Wave EigenVector
	return {alphaF(dens,B,a),alphaF(dens,B,a)*(V.x-sqrtf(cubeSoundFx(dens,B,a))),
		(alphaF(dens,B,a)*V.y)+(alphaS(dens,B,a)*betaY(B)*vAlfx(dens,B)),
		(alphaF(dens,B,a)*V.z)+(alphaS(dens,B,a)*betaZ(B)*vAlfx(dens,B)),
		0.f,
		(alphaS(dens,B,a)*sqrtf(cubeSoundFx(dens,B,a))*betaY(B))/(sqrtf(dens)),
		(alphaS(dens,B,a)*sqrtf(cubeSoundFx(dens,B,a))*betaZ(B))/(sqrtf(dens)),
		(alphaF(dens,B,a)*0.5f*dotProd(V,V))+minusAlphaHslow(dens,V,B,a)};
}

__device__ OctVec DMFEVec(Vec3d B){//Divergence of the Magnetic Field EigenVector
	return {0.f,
		0.f,
		0.f,
		0.f,
		1.f,
		0.f,
		0.f,
		B.x};
}


////////////////////////////////////////////////////
//Eigenvalues:

__device__ float EWEVal(Vec3d v){
	return v.x;
}

__device__ float plusAWEVal(float dens, Vec3d v, Vec3d B){
	return v.x + vAlfx(dens,B);
}

__device__ float minusAWEVal(float dens, Vec3d v, Vec3d B){
	return v.x - vAlfx(dens,B);
}

__device__ float plusSlowMAEVal(float dens, Vec3d v, Vec3d B, float a){
	return v.x + sqrtf(cubeSoundSx(dens,B,a));
}

__device__ float minusSlowMAEVal(float dens, Vec3d v, Vec3d B, float a){
	return v.x - sqrtf(cubeSoundSx(dens,B,a));
}

__device__ float plusFastMAEVal(float dens, Vec3d v, Vec3d B, float a){
	return v.x + sqrtf(cubeSoundFx(dens,B,a));
}

__device__ float minusFastMAEVal(float dens, Vec3d v, Vec3d B, float a){
	return v.x - sqrtf(cubeSoundFx(dens,B,a));
}

__device__ float DMFEVal(Vec3d v){
	return v.x;
}


///////////////////////////////////////////////////////
//Eigenvalues by Eigenvector Multiplication
/*
__device__ OctVec multEigenVec(float dens, Vec3d v, Vec3d B, float a){
	const OctVec EWE = EWEVec(v);
	const OctVec plusAWE = plusAWEVec(dens, v, B);
	const OctVec minusAWE = minusAWEVec(dens, v, B);
	const OctVec plusSlowMAE = plusSlowMAEVec(dens, v, B, a);
	const OctVec minusSlowMAE = minusSlowMAEVec(dens, v, B, a);
	const OctVec plusFastMAE = plusFastMAEVec(dens, v, B, a);
	const OctVec minusFastMAE = minusFastMAEVec(dens, v, B, a);
	const OctVec DMFE = DMFEVec(B);
	const float EigValEWE = EWEVal(v);
	const float EigValplusAWE = plusAWEVal(dens, v, B);
	const float EigValminusAWE = minusAWEVal(dens, v, B);
	const float EigValplusSlowMAE = plusSlowMAEVal(dens, v, B, a);
	const float EigValminusSlowMAE = minusSlowMAEVal(dens, v, B, a);
	const float EigValplusFastMAE = plusFastMAEVal(dens, v, B, a);
	const float EigValminusFastMAE = minusFastMAEVal(dens, v, B, a);
	const float EigValDMFE = DMFEVal(v);
	float densValues = 0.5f*(abs(EigValEWE)*EWE.dens + abs(EigValplusAWE)*plusAWE.dens + abs(EigValminusAWE)*minusAWE.dens +
			abs(EigValplusSlowMAE)*plusSlowMAE.dens + abs(EigValminusSlowMAE)*minusSlowMAE.dens + abs(EigValplusFastMAE)*plusFastMAE.dens +
			abs(EigValminusFastMAE)*minusFastMAE.dens + abs(EigValDMFE)*DMFE.dens);
	float momPxValues = 0.5f*(abs(EigValEWE)*EWE.momPx + abs(EigValplusAWE)*plusAWE.momPx + abs(EigValminusAWE)*minusAWE.momPx +
			abs(EigValplusSlowMAE)*plusSlowMAE.momPx + abs(EigValminusSlowMAE)*minusSlowMAE.momPx + abs(EigValplusFastMAE)*plusFastMAE.momPx +
			abs(EigValminusFastMAE)*minusFastMAE.momPx + abs(EigValDMFE)*DMFE.momPx);
	float momPyValues = 0.5f*(abs(EigValEWE)*EWE.momPy + abs(EigValplusAWE)*plusAWE.momPy + abs(EigValminusAWE)*minusAWE.momPy +
			abs(EigValplusSlowMAE)*plusSlowMAE.momPy + abs(EigValminusSlowMAE)*minusSlowMAE.momPy + abs(EigValplusFastMAE)*plusFastMAE.momPy +
			abs(EigValminusFastMAE)*minusFastMAE.momPy + abs(EigValDMFE)*DMFE.momPy);
	float momPzValues = 0.5f*(abs(EigValEWE)*EWE.momPz + abs(EigValplusAWE)*plusAWE.momPz + abs(EigValminusAWE)*minusAWE.momPz +
			abs(EigValplusSlowMAE)*plusSlowMAE.momPz + abs(EigValminusSlowMAE)*minusSlowMAE.momPz + abs(EigValplusFastMAE)*plusFastMAE.momPz +
			abs(EigValminusFastMAE)*minusFastMAE.momPz + abs(EigValDMFE)*DMFE.momPz);
	float BxValues = 0.5f*(abs(EigValEWE)*EWE.Bx + abs(EigValplusAWE)*plusAWE.Bx + abs(EigValminusAWE)*minusAWE.Bx +
			abs(EigValplusSlowMAE)*plusSlowMAE.Bx + abs(EigValminusSlowMAE)*minusSlowMAE.Bx + abs(EigValplusFastMAE)*plusFastMAE.Bx +
			abs(EigValminusFastMAE)*minusFastMAE.Bx + abs(EigValDMFE)*DMFE.Bx);
	float ByValues = 0.5f*(abs(EigValEWE)*EWE.By + abs(EigValplusAWE)*plusAWE.By + abs(EigValminusAWE)*minusAWE.By +
			abs(EigValplusSlowMAE)*plusSlowMAE.By + abs(EigValminusSlowMAE)*minusSlowMAE.By + abs(EigValplusFastMAE)*plusFastMAE.By +
			abs(EigValminusFastMAE)*minusFastMAE.By + abs(EigValDMFE)*DMFE.By);
	float BzValues = 0.5f*(abs(EigValEWE)*EWE.Bz + abs(EigValplusAWE)*plusAWE.Bz + abs(EigValminusAWE)*minusAWE.Bz +
			abs(EigValplusSlowMAE)*plusSlowMAE.Bz + abs(EigValminusSlowMAE)*minusSlowMAE.Bz + abs(EigValplusFastMAE)*plusFastMAE.Bz +
			abs(EigValminusFastMAE)*minusFastMAE.Bz + abs(EigValDMFE)*DMFE.Bz);
	float eValues = 0.5f*(abs(EigValEWE)*EWE.e + abs(EigValplusAWE)*plusAWE.e + abs(EigValminusAWE)*minusAWE.e +
			abs(EigValplusSlowMAE)*plusSlowMAE.e + abs(EigValminusSlowMAE)*minusSlowMAE.e + abs(EigValplusFastMAE)*plusFastMAE.e +
			abs(EigValminusFastMAE)*minusFastMAE.e + abs(EigValDMFE)*DMFE.e);
	return {densValues,
	momPxValues,
	momPyValues,
	momPzValues,
	BxValues,
	ByValues,
	BzValues,
	eValues
	};
}
*/
/*
__device__ OctVec LeftEigenVec(float dens, Vec3d v, Vec3d B, float a){
	const OctVec EWE = EWEVec(v);
	const OctVec plusAWE = plusAWEVec(dens, v, B);
	const OctVec minusAWE = minusAWEVec(dens, v, B);
	const OctVec plusSlowMAE = plusSlowMAEVec(dens, v, B, a);
	const OctVec minusSlowMAE = minusSlowMAEVec(dens, v, B, a);
	const OctVec plusFastMAE = plusFastMAEVec(dens, v, B, a);
	const OctVec minusFastMAE = minusFastMAEVec(dens, v, B, a);
	const OctVec DMFE = DMFEVec(B);
	float densValues = EWE.dens + plusAWE.dens + minusAWE.dens +
			plusSlowMAE.dens + minusSlowMAE.dens + plusFastMAE.dens +
			minusFastMAE.dens + DMFE.dens;
	float momPxValues = EWE.momPx + plusAWE.momPx + minusAWE.momPx +
			plusSlowMAE.momPx + minusSlowMAE.momPx + plusFastMAE.momPx +
			minusFastMAE.momPx + DMFE.momPx;
	float momPyValues = EWE.momPy + plusAWE.momPy + minusAWE.momPy +
			plusSlowMAE.momPy + minusSlowMAE.momPy + plusFastMAE.momPy +
			minusFastMAE.momPy + DMFE.momPy;
	float momPzValues = EWE.momPz + plusAWE.momPz + minusAWE.momPz +
			plusSlowMAE.momPz + minusSlowMAE.momPz + plusFastMAE.momPz +
			minusFastMAE.momPz + DMFE.momPz;
	float BxValues = EWE.Bx + plusAWE.Bx + minusAWE.Bx +
			plusSlowMAE.Bx + minusSlowMAE.Bx + plusFastMAE.Bx +
			minusFastMAE.Bx + DMFE.Bx;
	float ByValues = EWE.By + plusAWE.By + minusAWE.By +
			plusSlowMAE.By + minusSlowMAE.By + plusFastMAE.By +
			minusFastMAE.By + DMFE.By;
	float BzValues = EWE.Bz + plusAWE.Bz + minusAWE.Bz +
			plusSlowMAE.Bz + minusSlowMAE.Bz + plusFastMAE.Bz +
			minusFastMAE.Bz + DMFE.Bz;
	float eValues = (EWE.e + plusAWE.e + minusAWE.e +
			plusSlowMAE.e + minusSlowMAE.e + plusFastMAE.e +
			minusFastMAE.e + DMFE.e);
	return {densValues,
		momPxValues,
		momPyValues,
		momPzValues,
		BxValues,
		ByValues,
		BzValues,
		eValues};
}
*/
/////////////////////////////////////////////////////
//State, Source and Loss vectors


__device__ OctVec State(float dens, Vec3d v, Vec3d B, float p){
	const float e = 0.5f*((dens*dotProd(v,v)) + (2.f*p/(gamma-1.f)) + (dotProd(B,B)));
	return {dens,
		dens*v.x,
		dens*v.y,
		dens*v.z,
		B.x,
		B.y,
		B.z,
		e};
}

__device__ OctVec Source(double dens, Vec3d v, Vec3d vn, Vec3d B, double p, double dist){
	const double eta = (lambda/(mc*amu2kg)) * (kin/powf(cmPerM,3))/(vel_n) * rhowind;
	const float srcCoeff = 1.f*((mc*amu2kg) * Q * fi * (1.f) * expf(-dist))/
				(4.f*PI*rhowind*awind*lambda*lambda*powf(mPerkm,3)*dist*dist);
//	const float srcCoeff = 1.f;
//	double srcCoeff = 5.f*expf(-dist)/(dist*dist);
	return {srcCoeff,
		srcCoeff*(vn.x + (eta*dens*(vn.x - v.x))),
		srcCoeff*(vn.y + (eta*dens*(vn.y - v.y))),
		srcCoeff*(vn.z + (eta*dens*(vn.z - v.z))),
		0.f,
		0.f,
		0.f,
		srcCoeff*(0.5f*((vn.x*vn.x) + (eta*dens*(vn.x*vn.x - v.x*v.x)) - (3.f*eta*p)))};
}

__device__ double eTemp(double dist){
	double temp = 0.f;
	double realDist = dist*lambda;
	if (realDist <= powf(10.f, 3.2f)) temp = 100.f;
	if ((realDist > powf(10.f, 3.2f)) && (realDist <= powf(10.f, 3.84f)))
		temp = powf(10.f, 1.143f * __log10f(realDist) - 1.667f);
	if ((realDist > powf(10.f, 3.84f)) && (realDist <= powf(10.f, 4.f)))
		temp = powf(10.f, 10.965f * __log10f(realDist) - 39.3725f);
	if ((realDist > powf(10.f, 4.f)) && (realDist <= powf(10.f, 5.f)))
				temp = powf(10.f, 0.5135f * __log10f(realDist) + 2.4325f);
	if (realDist >= pow(10.f, 4.f)) temp = 1.e+5f;
	return temp;
}

__device__ OctVec Loss(double Dens, Vec3d v, double p, double dist){
	double recomb = 0.f;
	double expon = 0.f;
	const double edens = (nsw * msw) / mc;
	if (eTemp(dist) <= 200.f){
		recomb = recomb0 * sqrtf(300.f / eTemp(dist));
	} else {
		expon = 0.2553f - 0.1633f * __log10f(eTemp(dist));
		recomb = 2.342f * recomb0* powf(eTemp(dist), expon);
	}
	double	lossCoeff = recomb * (lambda / awind) * edens;
	return {lossCoeff * Dens,
		lossCoeff*Dens*v.x,
		lossCoeff*Dens*v.y,
		lossCoeff*Dens*v.z,
		0.f,
		0.f,
		0.f,
		lossCoeff*(0.5f*Dens*v.x + 1.5f*p)};
}


///////////////////////////////////////////////////////
//Define the State Difference

__device__ OctVec stateDiff(OctVec stateLeft, OctVec stateRight){
	return {(stateRight.dens - stateLeft.dens),
		(stateRight.momPx - stateLeft.momPx),
		(stateRight.momPy - stateLeft.momPy),
		(stateRight.momPz - stateLeft.momPz),
		(stateRight.Bx - stateLeft.Bx),
		(stateRight.By - stateLeft.By),
		(stateRight.Bz - stateLeft.Bz),
		(stateRight.e - stateLeft.e)
	};
}

__device__ double dotProdUnit(Vec3d X, float X0){
	return {X.x*X0 + X.y*X0 + X.z*X0};
}

__device__ OctVec FluxLeft(double dens, Vec3d v, Vec3d B, OctVec stateLeft){
	dens = stateLeft.dens;
	double vx = stateLeft.momPx/stateLeft.dens;
	double vy = stateLeft.momPy/stateLeft.dens;
	double vz = stateLeft.momPz/stateLeft.dens;
	double Bx = stateLeft.Bx;
	double By = stateLeft.By;
	double Bz = stateLeft.Bz;
	double dotprodUnit = (vx + vy + vz);
	double p = dens*(vx*vx + vy*vy + vz*vz);
	double e = 0.5f*(dens*(vx*vx + vy*vy + vz*vz) + 2.f*p/(gamma-1.f) + (Bx*Bx + By*By + Bz*Bz));
	return {dens*dotprodUnit,
		dens*vx*dotprodUnit + (p + 0.5f*Bx*Bx) - Bx*(Bx + By + Bz),
		dens*vy*dotprodUnit + (p + 0.5f*By*By) - By*(Bx + By + Bz),
		dens*vz*dotprodUnit + (p + 0.5f*Bz*Bz) - Bz*(Bx + By + Bz),
		vx*(Bx+By+Bz) - Bx*dotprodUnit,
		vy*(Bx+By+Bz) - By*dotprodUnit,
		vz*(Bx+By+Bz) - Bz*dotprodUnit,
		dotprodUnit*(e + p + 0.5f*(Bx*Bx + By*By + Bz*Bz)) -
		(Bx*vx + By*vy + Bz*vz)*(Bx + By + Bz)
	};
/*	return {dens*vx,
		dens*vx*vx + (p + 0.5f*Bx*Bx) - Bx*Bx,
		dens*vy*vy + (p + 0.5f*By*By) - By*Bx,
		dens*vz*vz + (p + 0.5f*Bz*Bz) - Bz*Bx,
		vx*Bx - Bx*vx,
		vy*By - By*vy,
		vz*Bz - Bz*vz,
		vx*(e + p + 0.5f*(Bx*Bx + By*By + Bz*Bz)) -
		(Bx*vx + By*vy + Bz*vz)*Bx
	};*/
}

__device__ OctVec FluxRight(double dens, Vec3d v, Vec3d B, OctVec stateRight){
	dens = stateRight.dens;
	double vx = stateRight.momPx/stateRight.dens;
	double vy = stateRight.momPy/stateRight.dens;
	double vz = stateRight.momPz/stateRight.dens;
	double Bx = stateRight.Bx;
	double By = stateRight.By;
	double Bz = stateRight.Bz;
	double dotprodUnit = (vx + vy + vz);
	double p = dens*(vx*vx + vy*vy + vz*vz);
	double e = 0.5f*(dens*(vx*vx + vy*vy + vz*vz) + 2.f*(p)/(gamma-1.f) + (Bx*Bx + By*By + Bz*Bz));
	return {dens*dotprodUnit,
		dens*vx*dotprodUnit + (p + 0.5f*Bx*Bx) - Bx*(Bx + By + Bz),
		dens*vy*dotprodUnit + (p + 0.5f*By*By) - By*(Bx + By + Bz),
		dens*vz*dotprodUnit + (p + 0.5f*Bz*Bz) - Bz*(Bx + By + Bz),
		vx*(Bx+By+Bz) - Bx*dotprodUnit,
		vy*(Bx+By+Bz) - By*dotprodUnit,
		vz*(Bx+By+Bz) - Bz*dotprodUnit,
		dotprodUnit*(e + p + 0.5f*(Bx*Bx + By*By + Bz*Bz)) -
		(Bx*vx + By*vy + Bz*vz)*(Bx + By + Bz)
	};
/*	return {dens*vx,
		dens*vx*vx + (p + 0.5f*Bx*Bx) - Bx*Bx,
		dens*vy*vy + (p + 0.5f*By*By) - By*Bx,
		dens*vz*vz + (p + 0.5f*Bz*Bz) - Bz*Bx,
		vx*Bx - Bx*vx,
		vy*By - By*vy,
		vz*Bz - Bz*vz,
		vx*(e + p + 0.5f*(Bx*Bx + By*By + Bz*Bz)) -
		(Bx*vx + By*vy + Bz*vz)*Bx
	};*/
}

