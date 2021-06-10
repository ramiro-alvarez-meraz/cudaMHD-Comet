/*
 * render_methods.cu

 *
 *  Created on: Sep 20, 2019
 *      Author: ramiro
 */

#include "struct.h"
#include "device_funcs.cuh"
#include "helper_math.h"
#include <stdio.h>
#include "const.h"
#include "params.h"


__device__ uchar4 sliceShader(double *d_dens, double *d_momPx,
		double *d_momPy, double *d_momPz, double *d_Bx, double *d_By,
		double *d_Bz, double *d_e, int3 volSize, Ray boxRay,
  float gain, float dist, float3 norm, int id) {
  float t;
//  uchar4 shade = make_uchar4(96, 0, 192, 0); // background value
  uchar4 shade = make_uchar4(100, 0, 200, 0); // background value
  //uchar4 shade = make_uchar4(0, 0, 0, 0);
  if (rayPlaneIntersect(boxRay, norm, dist, &t)) {
    float sliceDens = density(d_dens, d_momPx, d_momPy, d_momPz,
    		d_Bx, d_By, d_Bz, d_e, volSize, paramRay(boxRay, t),
    		id);
 //   shade = make_uchar4(48, clip(10.f * (5.0f + gain) * sliceDens), 96, 255);
//    const float maxScale = 7.f;
//    if (sliceDens <= 1.f) sliceDens = 1.f;
/*    shade = make_uchar4(clip(255.f - (sliceDens)),
    		192,
    		clip(sliceDens), 255);*/
    if (id == 0){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/6.f)),
    		192,
    		clip(255.f*sliceDens/6.f),
    		255);}
    if (id == 1){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/5.f)),
    		192,
    		clip(255.f*sliceDens/5.f),
    		255);}
    if (id == 2){
    shade = make_uchar4(clip(255.f*(1.f - (sliceDens/IMF))),
    		192,
    		clip(255.f*(sliceDens/IMF)),
    		255);}
    if (id == 3){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/60.f)),
    		192,
    		clip(255.f*sliceDens/60.f),
    		255);}
    if (id == 4){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/5.f)),
    		192,
    		clip(255.f*sliceDens/5.f),
    		255);}
    if (id == 5){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/1.f)),
    		192,
    		clip(255.f*sliceDens/1.f),
    		255);}
    if (id == 6){
    shade = make_uchar4(clip(255.f*(1.f - sliceDens/10.f)),
    		192,
    		clip(255.f*sliceDens/10.f),
    		255);}

//    if (id == 2) {
//    	shade = make_uchar4(48, clip(1.f * (5.0f + gain) * sliceDens),
//    	                        96, 255);
//    }
//   printf("%f\t",sliceDens);
  }
  return shade;
}
/*
__device__ uchar4 volumeRenderShader(float *d_dens, Vec3d *d_momP, Vec3d *d_B, float *d_e, int3 volSize,
  Ray boxRay, float threshold, int numSteps, int id) {
  uchar4 shade = make_uchar4(96, 0, 192, 0); // background value
  const float dt = 1.f / numSteps;
  const float len = length(boxRay.d) / numSteps;
  float accum = 0.f;
  float3 pos = boxRay.o;
  //float val = density(d_dens, volSize, pos);
  float val = density(d_dens, d_momP, d_B, d_e, volSize, pos, id);
  for (float t = 0.f; t<1.f; t += dt) {
   // if (val - threshold < 0.f) accum += (fabsf(val - threshold))*len;
	if (val > 0.f) accum += (val-threshold)*len;
    pos = paramRay(boxRay, t);
    val = density(d_dens, d_momP, d_B, d_e, volSize, pos, id);
    //printf("%f\n",pos);
  }
  if (clip(accum) > 0.f) shade.x = clip(accum);
  return shade;
}

__device__ uchar4 rayCastShader(float *d_dens, Vec3d *d_momP, Vec3d *d_B, float *d_e, int3 volSize,
  Ray boxRay, float dist, int id) {
  uchar4 shade = make_uchar4(96, 0, 192, 0);
  float3 pos = boxRay.o;
  float len = length(boxRay.d);
  float t = 0.0f;
  float f = density(d_dens, d_momP, d_B, d_e, volSize, pos, id);
  while (f > dist + EPS && t < 1.0f) {
    f = density(d_dens, d_momP, d_B, d_e, volSize, pos, id);
    t += (f - dist) / len;
    pos = paramRay(boxRay, t);
    f = density(d_dens, d_momP, d_B, d_e, volSize, pos, id);
  }
  if (t < 1.f) {
    const float3 ux = make_float3(1, 0, 0), uy = make_float3(0, 1, 0),
                 uz = make_float3(0, 0, 1);
    float3 grad = {(density(d_dens, d_momP, d_B, d_e, volSize, pos + EPS*ux, id) -
                    density(d_dens, d_momP, d_B, d_e, volSize, pos, id))/EPS,
                   (density(d_dens, d_momP, d_B, d_e, volSize, pos + EPS*uy, id) -
                   density(d_dens, d_momP, d_B, d_e, volSize, pos, id))/EPS,
                   (density(d_dens, d_momP, d_B, d_e, volSize, pos + EPS*uz, id) -
                   density(d_dens, d_momP, d_B, d_e, volSize, pos, id))/EPS};
    float intensity = -dot(normalize(boxRay.d), normalize(grad));
    shade = make_uchar4(255 * intensity, 0, 0, 255);
  }
  return shade;
}
*/
