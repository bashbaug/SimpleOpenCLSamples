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

#include "nanoglut.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)

HWND nglut_hWnd = NULL;
HDC nglut_hDC = NULL;
HGLRC nglut_hRC = NULL;

#elif defined(__linux__)

Display *nglut_hDisplay = NULL;
Window nglut_hWin = 0;
GLXContext nglut_hRC = NULL;

#else
#error Unknown OS!
#endif

static int nglut_width = 300;
static int nglut_height = 300;

void nglutInitWindowSize(int width, int height)
{
    nglut_width = width;
    nglut_height = height;
}

static void defaultReshapeFunc(
    int width,
    int height )
{
}

static void defaultDisplayFunc(void)
{
}

static void defaultKeyboardFunc(
    unsigned char key,
    int x,
    int y )
{
}

static void defaultIdleFunc()
{
}

static void (*MyReshapeFunc)(int width, int height) = defaultReshapeFunc;
static void (*MyDisplayFunc)(void) = defaultDisplayFunc;
static void (*MyKeyboardFunc)(unsigned char key, int x, int y) = defaultKeyboardFunc;
static void (*MyIdleFunc)() = defaultIdleFunc;

void nglutReshapeFunc(
    void (*func)(int width, int height))
{
    MyReshapeFunc = func;
}

void nglutDisplayFunc(
    void (*func)(void))
{
    MyDisplayFunc = func;
}

void nglutKeyboardFunc(
    void (*func)(unsigned char key, int x, int y))
{
    MyKeyboardFunc = func;
}

void nglutIdleFunc(
    void (*func)(void))
{
    MyIdleFunc = func;
}

#if defined(_WIN32)

static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch( msg )
    {
    case WM_KEYDOWN:
        MyKeyboardFunc(
            (unsigned char)wParam,
            0,
            0 );
        break;

    case WM_SIZE:
        MyReshapeFunc(
            LOWORD( lParam ),
            HIWORD( lParam ) );
        break;
    }

    return DefWindowProc( hWnd, msg, wParam, lParam );
}

bool nglutCreateWindow(const char* title)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wc;
    DWORD dwStyle;
    int bpp;

    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszMenuName = NULL;
    wc.lpszClassName = "nanoglut";

    RegisterClass(&wc);
    
    dwStyle = WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_OVERLAPPED | WS_VISIBLE;

    if((nglut_hWnd = CreateWindow("nanoglut", title, 
        dwStyle,
        0, 0, 
        (int)(nglut_width  + (GetSystemMetrics(SM_CXSIZEFRAME)-1)*2), 
        (int)(nglut_height + (GetSystemMetrics(SM_CYSIZEFRAME)-1)*2 + GetSystemMetrics(SM_CYMENU)),
        NULL, 
        NULL, 
        hInstance,
        NULL)) == NULL)
    {
        MessageBox(NULL, "CreateWindow failed (returned NULL).\n", "ERROR!", MB_OK);
        return false;
    }

    PIXELFORMATDESCRIPTOR pfd;
    int pixelformat;

    nglut_hDC = GetDC(nglut_hWnd);
    bpp = GetDeviceCaps(nglut_hDC, BITSPIXEL);

    ZeroMemory(&pfd, sizeof(pfd));
    pfd.nSize = sizeof( pfd );
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cAlphaBits = (bpp == 32) ?  8 : 0;
    pfd.cColorBits = (bpp == 32) ? 24 : bpp;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;

    if ((pixelformat = ChoosePixelFormat(nglut_hDC, &pfd)) == 0) {
        MessageBox(NULL, "ChoosePixelFormat failed (returned 0).\n", "ERROR!", MB_OK);
        return false;
    }

    if (DescribePixelFormat(nglut_hDC, pixelformat, sizeof(pfd), &pfd) == 0) {
        MessageBox(NULL, "DescribePixelFormat failed (returned 0).\n", "ERROR!", MB_OK);
        return false;
    }

    if (SetPixelFormat(nglut_hDC, pixelformat, &pfd) == FALSE) {
        MessageBox(NULL, "SetPixelFormat failed (returned FALSE).\n", "ERROR!", MB_OK);
        return false;
    }

    if ((nglut_hRC = wglCreateContext(nglut_hDC)) == NULL) {
        MessageBox(NULL, "wglCreateContext failed (returned NULL).\n", "ERROR!", MB_OK);
        return false;
    }

    if (wglMakeCurrent(nglut_hDC, nglut_hRC) == FALSE) {
        MessageBox(NULL, "wglMakeCurrent failed (returned FALSE).\n", "ERROR!", MB_OK);
        return false;
    }

    return true;
}

void nglutPostRedisplay()
{
    MyDisplayFunc();
    InvalidateRect(nglut_hWnd, NULL, FALSE);
}

void nglutSwapBuffers()
{
    SwapBuffers(nglut_hDC);
}

void nglutMainLoop()
{
    bool done = false;
    MSG msg;

    while (!done) {
        while (PeekMessage( &msg, 0, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                done = true;
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        MyIdleFunc();
    }
}

void nglutLeaveMainLoop()
{
    PostQuitMessage(0);
}

void nglutShutdown()
{
    wglMakeCurrent(NULL, NULL);

    wglDeleteContext(nglut_hRC);
    nglut_hRC = NULL;

    ReleaseDC(nglut_hWnd, nglut_hDC);
    nglut_hDC = NULL;
    
    DestroyWindow(nglut_hWnd);
    nglut_hWnd = NULL;
}

#elif defined(__linux__)

#define NGLUT_DEBUG(_msg)
//#define NGLUT_DEBUG(_msg) fprintf(stderr, _msg);

bool nglutCreateWindow(const char* title)
{
    NGLUT_DEBUG("Enter: nglutCreateWindow()\n");

    XSetWindowAttributes    xAttr;

    // open display
    NGLUT_DEBUG("XOpenDisplay()\n");
    nglut_hDisplay = XOpenDisplay(0);
    if (nglut_hDisplay == NULL) {
        NGLUT_DEBUG("XOpenDisplay() returned NULL.\n");
        return false;
    }
    
    // choose visual
    NGLUT_DEBUG("glXChooseVisual()\n");
    int glAttr[] = {
        GLX_RGBA,
        GLX_DOUBLEBUFFER,
        GLX_DEPTH_SIZE,   24,
        GLX_STENCIL_SIZE, 8,
        GLX_RED_SIZE,     8,
        GLX_GREEN_SIZE,   8,
        GLX_BLUE_SIZE,    8,
        GLX_ALPHA_SIZE,   8,
        None
    };
    XVisualInfo *vi = glXChooseVisual( nglut_hDisplay, DefaultScreen(nglut_hDisplay), glAttr );
    if (vi == NULL) {
        NGLUT_DEBUG("glXChooseVisual() returned NULL.\n");
        return false;
    }
  
    // create context
    NGLUT_DEBUG("glXCreateContext()\n");
    nglut_hRC = glXCreateContext( nglut_hDisplay, vi, 0, GL_TRUE );
    if (nglut_hRC == NULL) {
        NGLUT_DEBUG("glXCreateContext() returned NULL.\n");
        return false;
    }

    // set up colormap and border pixel
    NGLUT_DEBUG("XCreateColormap()\n");
    Colormap cmap = XCreateColormap(
        nglut_hDisplay,
        RootWindow(nglut_hDisplay, vi->screen),
        vi->visual,
        AllocNone);
    xAttr.colormap=cmap;
    xAttr.border_pixel=0;

    // set up event mask
    xAttr.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | StructureNotifyMask;

    // create window
    NGLUT_DEBUG("XCreateWindow()\n");
    nglut_hWin = XCreateWindow(
        nglut_hDisplay,
        RootWindow(nglut_hDisplay, vi->screen),
        0, 0,   // x, y
        nglut_width, nglut_height,
        0,
        vi->depth,
        InputOutput,
        vi->visual,
        CWBorderPixel | CWColormap | CWEventMask,
        &xAttr);

    // set up deletion message
    NGLUT_DEBUG("XInternAtom()\n");
    Atom wmDelete = XInternAtom(nglut_hDisplay, "WM_DELETE_WINDOW", true);
    XSetWMProtocols(nglut_hDisplay, nglut_hWin, &wmDelete, 1);

    // a few additional properties
    NGLUT_DEBUG("XSetStandardProperties()\n");
    XSetStandardProperties(nglut_hDisplay, nglut_hWin, title, title, None, NULL, 0, NULL);

    // map window
    NGLUT_DEBUG("XMapRaised()\n");
    XMapRaised(nglut_hDisplay, nglut_hWin);
  
    // make the rendering context current
    NGLUT_DEBUG("glXMakeCurrent()\n");
    if (!glXMakeCurrent( nglut_hDisplay, nglut_hWin, nglut_hRC)) {
        NGLUT_DEBUG("glXMakeCurrent() returned false.\n");
        return false;
    }

    NGLUT_DEBUG("Exit: nglutCreateWindow()\n");
    return true;
}

void nglutPostRedisplay()
{
    MyDisplayFunc();
    // TODO?
    //InvalidateRect(nglut_hWnd, NULL, FALSE);
}

void nglutSwapBuffers()
{
    glXSwapBuffers(nglut_hDisplay, nglut_hWin);
}

void nglutMainLoop()
{
    NGLUT_DEBUG("Enter: nglutMainLoop()\n");

    bool done = false;

    while (!done) {
        while( XPending(nglut_hDisplay) > 0 ) {
            XEvent event;
            XNextEvent(nglut_hDisplay, &event);
            switch(event.type) {
            case Expose:
                // Guess this is called when the window gets focus?
                break;
            case ConfigureNotify:
                // Window resize, among other things.
                {
                    XConfigureEvent xce = event.xconfigure;
                    MyReshapeFunc(
                        xce.width,
                        xce.height );
                }
                break;
            case ButtonPress:
                // Mouse button press.  Do nothing for now.
                break;
            case KeyPress:
                // Key is pressed.
                {
                    KeySym  k = XLookupKeysym( &event.xkey, 0 );
                    unsigned char c;
                    switch( k )
                    {
                    case XK_Escape:     c = 27;     break;
                    case XK_space:      c = ' ';    break;
                    default:
                        c = (unsigned char)k;
                        c = toupper(c);
                        break;
                    }
                    MyKeyboardFunc(
                        c,
                        0,
                        0 );
                }
                break;
            case ClientMessage:
                // Windows closed.
                if (*XGetAtomName(nglut_hDisplay, event.xclient.message_type) == *"WM_PROTOCOLS") {
                    done = true;
                }
                break;
            default:
                break;
            }
        }

        MyIdleFunc();
    }

    NGLUT_DEBUG("Exit: nglutMainLoop()\n");
}

void nglutLeaveMainLoop()
{
    XEvent event = { 0 };
    event.xclient.type = ClientMessage;
    event.xclient.window = nglut_hWin;
    event.xclient.message_type = XInternAtom(nglut_hDisplay, "WM_PROTOCOLS", true);
    event.xclient.format = 32;
    event.xclient.data.l[0] = XInternAtom(nglut_hDisplay, "WM_DELETE_WINDOW", false);
    event.xclient.data.l[1] = CurrentTime;
    XSendEvent(nglut_hDisplay, nglut_hWin, False, NoEventMask, &event);
}

void nglutShutdown()
{
    NGLUT_DEBUG("Enter: nglutShutdown()\n");

    if (nglut_hRC) {
        NGLUT_DEBUG("glxMakeCurrent( NULL )\n");
        glXMakeCurrent( nglut_hDisplay, None, NULL );
        NGLUT_DEBUG("glXDestroyContext()\n");
        glXDestroyContext( nglut_hDisplay, nglut_hRC );
        nglut_hRC = NULL;
    }

    if (nglut_hWin) {
        NGLUT_DEBUG("glXDestroyWindow()\n");
        XDestroyWindow( nglut_hDisplay, nglut_hWin );
        nglut_hWin = 0;
    }

    if (nglut_hDisplay) {
        NGLUT_DEBUG("XCloseDisplay()\n");
        XCloseDisplay( nglut_hDisplay );
        nglut_hDisplay = NULL;
    }

    NGLUT_DEBUG("Exit: nglutShutdown()\n");
}

#endif
