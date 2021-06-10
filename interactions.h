/*
 * interactions.h
 *
 *  Created on: Aug 27, 2019
 *      Author: ramiro
 */

#ifndef INTERACTIONS_H_
#define INTERACTIONS_H_
#include "struct.h"
#include "const.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector_types.h>

int time = 0;
int id = 0; // 0 = mass density, 1 = (mean) linear momentum, 2 = (mean) magnetic field,
//3 = internal energy density, 4 = gyroradius, 5 = x-velocity component, 6 = ENAs production
int method = 1; // 0 = volumeRender, 1 = slice, 2 = raycast
const int3 volSize = {NX, NY, NZ}; // pointer to device array for storing volume data
double *d_dens; 		// pointer to device array for storing density data
double *d_momPx;		// pointer to device array for storing momentum data
double *d_momPy;		// pointer to device array for storing momentum data
double *d_momPz;		// pointer to device array for storing momentum data
double *d_Bx;		// pointer to device array for storing magnetic data
double *d_By;		// pointer to device array for storing magnetic data
double *d_Bz;		// pointer to device array for storing magnetic data
double *d_e;			// pointer to device array for storing energy density data
double zs = NZ; // distance from origin to source
double dist = 0.f, theta = 0.f, threshold = 0.f;

void mymenu(int value) {
  switch (value) {
  case 0: return;
  case 1: id = 0; break; // mass density
  case 2: id = 1; break; // (mean) linear momentum
  case 3: id = 2; break; // (mean) magnetic field
  case 4: id = 3; break; // internal energy density
  case 5: id = 4; break; // gyroradius
  case 6: id = 5; break; // x-velocity component
  case 7: id = 6; break; // ENAs production
  }
  glutPostRedisplay();
}

void idle(void) {
    ++time;
	glutPostRedisplay();
}

void createMenu() {
  glutCreateMenu(mymenu); // Physical parameter selection menu
  glutAddMenuEntry("Physical Parameter Selector:", 0); // menu title
  glutAddMenuEntry("mass density", 1);
  glutAddMenuEntry("(mean) linear momentum", 2);
  glutAddMenuEntry("(mean) magnetic field", 3);
  glutAddMenuEntry("internal energy density", 4);
  glutAddMenuEntry("gyroradius", 5);
  glutAddMenuEntry("x-velocity", 6);
  glutAddMenuEntry("ENAs production", 7);
  glutAttachMenu(GLUT_RIGHT_BUTTON); // right-click for menu
}

void keyboard(unsigned char key, int x, int y) {
  if (key == '+') zs -= DELTA; // move source closer (zoom in)
  if (key == '-') zs += DELTA; // move source farther (zoom out)
  if (key == 'd') --dist; // decrease slice distance
  if (key == 'D') ++dist; // increase slice distance
  if (key == 'z') zs = NZ, theta = 0.f, dist = 0.f; // reset values
//  if (key == 'v') method = 0; // volume render
  if (key == 'f') method = 1; // slice
//  if (key == 'r') method = 2; // raycast
  if (key == 't') id = 0; // mass density
  if (key == 'g') id = 1; // (mean) linear momentum
  if (key == 'b') id = 2; // (mean) magnetic field
  if (key == 'y') id = 3; // internal energy density
  if (key == 'h') id = 4; // gyroradius
  if (key == 'n') id = 5; // x-velocity
  if (key == 'u') id = 6; // x-velocity
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
  if (key == GLUT_KEY_LEFT) theta -= PI/8.f; // rotate left
  if (key == GLUT_KEY_RIGHT) theta += PI/8.f; // rotate right
  if (key == GLUT_KEY_UP) threshold += 1.f; // inc threshold (thick)
  if (key == GLUT_KEY_DOWN) threshold -= 1.f; // dec threshold (thin)
  glutPostRedisplay();
}

void printInstructions() {
  printf("3D Comet Simulator\n"
         "Controls:\n"
//         "Volume render mode (not working)            : v\n"
         "Slice render mode                           : f\n"
//         "Raycast mode (not working)                  : r\n"
         "Zoom out/in                                 : -/+\n"
         "Rotate view                                 : left/right\n"
         "Decr./Incr. Offset (intensity in slice mode): down/up\n"
         "Decr./Incr. distance (only in slice mode)   : d/D\n"
         "Right-click for the physical parameter menu\n"
         "Mass density                                : t\n"
         "Mean linear momentum (default)              : g\n"
         "Mean magnetic field                         : b\n"
         "Internal energy density                     : y\n"
         "Gyroradius                                  : h\n"
		 "x-velocity component                        : n\n"
		 "ENAs production                             : u\n"
		 "Reset geometric parameters                  : z\n");
}

#endif /* INTERACTIONS_H_ */
