/*
 * struct.h
 *
 *  Created on: Oct 10, 2019
 *      Author: ramiro
 */

#ifndef STRUCT_H_
#define STRUCT_H_

struct uchar4;
struct int3;
struct float3;
struct float4;
struct double3;
struct double4;

typedef struct {
double dens, momPx, momPy, momPz, Bx, By, Bz, e;
} EigVec;


typedef struct {
double dens, momPx, momPy, momPz, Bx, By, Bz, e;
} OctVec;


typedef struct {
double x, y, z;
} Vec3d;

// struct BC that contains all the boundary conditions
typedef struct {
//  float rad; // radius of comet
	double dens_ext;
	double momPx_ext, momPy_ext, momPz_ext;
	double Bx_ext, By_ext, Bz_ext; // external magnetic field
	double e_ext;
} BC;


#endif

