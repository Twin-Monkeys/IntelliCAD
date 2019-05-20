#pragma once

#include "stdafx.h"
#include "Pixel.h"
#include <GL/glew.h>
#include <GL/wglew.h>
#include <cuda_gl_interop.h>

class CRenderingView : public CView
{
	DECLARE_MESSAGE_MAP()

public:
	/* member function */
	virtual void OnDraw(CDC* /*pDC*/);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);

protected:
	/* member function */
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) = 0;

private:
	/* member function */
	void __createDeviceBuffer(const int width, const int height);
	void __deleteDeviceBuffer();

	/* member variable */
	HDC __hDeviceContext = nullptr;
	HGLRC __hRenderingContext = nullptr;
	CSize __screenSize;
	GLuint __bufferObject = 0;
	cudaGraphicsResource* __pCudaRes = nullptr;

public:
	void render();
};