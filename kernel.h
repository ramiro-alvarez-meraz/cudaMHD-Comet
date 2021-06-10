/*
 * kernel.h
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#ifndef KERNEL_H_
#define KERNEL_H_


void kernelLauncher(uchar4 *d_out, double *d_dens, double *d_momPx, double *d_momPy, double *d_momPz,
		double *d_Bx, double *d_By,double *d_Bz, double *d_e, float time, int w, int h, int3 volSize, int method,
	float zs, float theta, float threshold, float dist, int id);

void resetKernelLauncher(double *d_dens, double *d_momPx, double *d_momPy, double *d_momPz,
		double *d_Bx, double *d_By, double *d_Bz, double *d_e, int3 volSize, int id);

void evolutiveKernelLauncher(double *d_dens, double *d_momPx, double *d_momPy, double *d_momPz,
		double *d_Bx, double *d_By, double *d_Bz, double *d_e, int time, int3 volSize, int id);

void writeData(int output, double *h_dens, double *h_momPx,
		double *h_momPy, double *h_momPz, double *h_Bx, double *h_By,
		double *h_Bz, double *h_e, int3 volSize);

#endif /* KERNEL_H_ */
