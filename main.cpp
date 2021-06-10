/*
 * main.cpp
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#include "interactions.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;


void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
  evolutiveKernelLauncher(d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz,
		  d_e, time, volSize, id);
  kernelLauncher(d_out, d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz,
		  d_e, time, W, H, volSize, method, zs, theta, threshold, dist, id);
//  evolutiveKernelLauncher(d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz,
//		  d_e, time, volSize, id);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
  char title[128];
  if (id == 0){
	  sprintf(title, "Comet Simulator : log10(mass density/rhowind) = 0 - 6,  time =%4d,"
	  	  	  " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 1){
	  sprintf(title, "Comet Simulator : log10(mean momentum/(amu*km/s)) = 0 - 5,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 2){
	  sprintf(title, "Comet Simulator : mean magnetic[nT] = 0 - 6,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 3){
	  sprintf(title, "Comet Simulator : pressure/(rhowind*awind^2) = 0 - 60,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 4){
	  sprintf(title, "Comet Simulator : log10(gyroradius/(km)) = 0 - 5,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 5){
	  sprintf(title, "Comet Simulator : log10(x-velocity/awind = 0 - 1,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  if (id == 6){
	  sprintf(title, "Comet Simulator : ENAs production[] = ,  time =%4d,"
	          " dist = %.1f, theta = %.1f", time, dist, theta);
  }
  glutSetWindowTitle(title);
}


void draw_texture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
    GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}


void display() {
  render();
  draw_texture();
  glutSwapBuffers();
}


void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(W, H);
  glutCreateWindow("Comet Simulator");
#ifndef __APPLE__
  glewInit();
#endif
}


void initPixelBuffer() {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, W*H*sizeof(GLubyte)* 4, 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                               cudaGraphicsMapFlagsWriteDiscard);
}


void allocParam() {
	cudaMalloc(&d_dens, NX*NY*NZ*sizeof(double)); // 3D dens data
	cudaMalloc(&d_momPx, NX*NY*NZ*sizeof(double)); // 3D momemtum data
	  cudaMalloc(&d_momPy, NX*NY*NZ*sizeof(double)); // 3D momemtum data
	  cudaMalloc(&d_momPz, NX*NY*NZ*sizeof(double)); // 3D momemtum data
	  cudaMalloc(&d_Bx, NX*NY*NZ*sizeof(double)); // 3D magnetic data
	  cudaMalloc(&d_By, NX*NY*NZ*sizeof(double)); // 3D magnetic data
	  cudaMalloc(&d_Bz, NX*NY*NZ*sizeof(double)); // 3D magnetic data
	  cudaMalloc(&d_e, NX*NY*NZ*sizeof(double)); // 3D energy data
}


void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
  cudaFree(d_dens);
  cudaFree(d_momPx);
  cudaFree(d_momPy);
  cudaFree(d_momPz);
  cudaFree(d_Bx);
  cudaFree(d_By);
  cudaFree(d_Bz);
  cudaFree(d_e);
}





int main(int argc, char** argv) {
  allocParam();
  resetKernelLauncher(d_dens, d_momPx, d_momPy, d_momPz, d_Bx, d_By, d_Bz, d_e,	volSize, id);
  printInstructions();
  initGLUT(&argc, argv);
  createMenu();
  gluOrtho2D(0, W, H, 0);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(handleSpecialKeypress);
  glutIdleFunc(idle);
  glutDisplayFunc(display);
  initPixelBuffer();
  glutMainLoop();
  atexit(exitfunc);

  return 0;
}
