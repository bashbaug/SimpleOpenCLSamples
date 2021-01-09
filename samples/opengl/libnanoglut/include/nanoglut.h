/*
// Copyright (c) 2021 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/

#pragma once

#if defined(_WIN32)

#include <windows.h>
extern HWND nglut_hWnd;
extern HDC nglut_hDC;
extern HGLRC nglut_hRC;

#elif defined(__linux__)

#include <GL/glx.h>
#include <X11/Xlib.h>
extern Display *nglut_hDisplay;
extern Window nglut_hWin;
extern GLXContext nglut_hRC;

#else
#error Unknown OS!
#endif

#include "GL/gl.h"

void nglutInitWindowSize(int width, int height);

// Note: there is no glutInitDisplayMode (yet)!

void nglutReshapeFunc(
    void (*func)(int width, int height));
void nglutDisplayFunc(
    void (*func)(void));
void nglutKeyboardFunc(
    void (*func)(unsigned char key, int x, int y));

bool nglutCreateWindow(const char* title);

void nglutPostRedisplay();
void nglutSwapBuffers();

void nglutMainLoop();
void nglutLeaveMainLoop();

// Note: this function does not exist in freeglut or openglut.
void nglutShutdown();
