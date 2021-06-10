/*
 * params.h

 *
 *  Created on: March 11, 2020
 *      Author: ramiro
 */
#ifndef PARAMS_H_
#define PARAMS_H_

#define	Q  		  7.e+29f 	//cometary gas production rate [molecules/s] (Giotto Observations)
#define Rn 		 10.f		//Radius of comet nucleus [km]
#define fi 		  1.f	    //Ratio between ionization time and free streaming ionization by solar wind
#define mc 		 17.f    	//Mean molecular mass of cometary ions[amu]
//#define lambda    1.0e+3f    //local ionization length [km]
//#define lambda    1.0e+4f    //local ionization length [km]
#define lambda    3.03e+4f    //local ionization length [km]
//#define lambda    1.0e+5f    //local ionization length [km]
//#define lambda    3.03e+5f    //local ionization length [km]
//#define lambda    1.e+6f    //local ionization length [km]
#define kin		  1.7e-9f	//Ion-neutral momentum transfer collision rate[cm^3/s]
#define rhowind   1.34e-20f //mass density in solar wind [kg/m^3]
#define solWind 371.f		//Solar wind speed [km/s]
#define awind    37.1f 		//sound speed in solar wind [km/s]
#define Tsw       1.e+5f	//Solar wind temperature [K]
#define recomb0   7.e-7f 	//recombination rate [K cm^3/s]
#define nsw       8.f		//solar wind number density [1/cm^3]
#define qecsH    3.e-19f	//charge exchange cross section between protons and hydrogen[m^2]
#define msw       1.f		//mean molecular mass in solar wind [amu]
#define vel_n     1.f		//terminal gas velocity of neutral molecules[km/s]
#define Mach     10.f		//Solar wind Match number
#define MachA    10.f		//Solar wind Alfvenic Mach number
#define IMF	      4.81f		//Interplanetary magnetic field [nT]
//#define IMF	      6.f		//Interplanetary magnetic field [nT]
#define mu		  1.25663706212e-6f	//magnetic permeability in vacuum[H/m]
#define IMFang PI/4.f		//IMF angle PI/4 [radian]
#define dt 		  50000.e+0f	//delta time
#define timestep  10.	//time between outputs
#define gamma 	  1.66666667f
#define LEN		  2.e+6f	//the weight of the box is 2*LEN
//#define LEN		  2.e+5f	//the weight of the box is 2*LEN
#define W 1000
#define H 1000
#define NX 128
#define NY 128
#define NZ 128
#define TX_2D 16
#define TY_2D 16
#define TX 4
#define TY 4
#define TZ 4
#define DELTA 10 // pixel increment for arrow keys
#define NUMSTEPS 50
#define RAD 1
#define EPS 0.01f

#endif
