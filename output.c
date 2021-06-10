/*
 * output.c
 *
 *  Created on: Mar 23, 2020
 *      Author: ramiro
 */

#include "helper_math.h"
#include <math.h>
#include <math_constants.h>
#include "const.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>



void writeData(int time, double *h_dens, double *h_momPx,
		double *h_momPy, double *h_momPz, double *h_Bx, double *h_By,
		double *h_Bz, double *h_e) {
	int output;
	double param, fractpart, intpart;
	param = float(time)/timestep;
	fractpart = modf (param , &intpart);
	if (fractpart == 0.f) {
		output = int(param);
		FILE *fp;
		char name[80];
		// write data dens
		sprintf (name, "out1/dens%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_dens, sizeof(double), NX*NY*NZ, fp);

		// write data momPx
		sprintf (name, "out1/momPx%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_momPx, sizeof(double), NX*NY*NZ, fp);

		// write data momPy
		sprintf (name, "out1/momPy%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_momPy, sizeof(double), NX*NY*NZ, fp);

		// write data momPz
		sprintf (name, "out1/momPz%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_momPz, sizeof(double), NX*NY*NZ, fp);

		// write data Bx
		sprintf (name, "out1/Bx%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_Bx, sizeof(double), NX*NY*NZ, fp);

		// write data By
		sprintf (name, "out1/By%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_By, sizeof(double), NX*NY*NZ, fp);

		// write data Bz
		sprintf (name, "out1/Bz%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_Bz, sizeof(double), NX*NY*NZ, fp);

		// write data e
		sprintf (name, "out1/e%d.dat", output);
		fp = fopen(name, "w");
		fwrite (h_e, sizeof(double), NX*NY*NZ, fp);

		fclose(fp);
	}
}
