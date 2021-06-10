/*
 * output.cu
 *
 *  Created on: Aug 29, 2019
 *      Author: ramiro
 */

#include <stdio.h>

__global__
void writeDisk(){
  FILE *output;
  char name[256];
  fprintf(output,"%d\t%d\t%d\t%d\t%d\t%d\t%d\t\n",dx,dy,dy,dens,vx,vy,vz);
}
