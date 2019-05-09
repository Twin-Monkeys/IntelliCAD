#pragma once

#include "stdafx.h"

#include <GL/glew.h>
#include <GL/wglew.h>
#include <cuda_gl_interop.h>
#include "Pixel.h"

class CRenderingView : public CView
{
public:
	virtual void OnDraw(CDC* /*pDC*/);

private:
	CPoint __prevPos;
	HDC __hDeviceContext = nullptr;
	HGLRC __hRenderingContext = nullptr;

	CSize __screenSize;
	GLuint __bufferObject = 0;
	cudaGraphicsResource* __pCudaRes = nullptr;

	void __createDeviceBuffer(const int width, const int height);
	void __deleteDeviceBuffer();
	void __render();

protected:
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) = 0;

public:
	DECLARE_MESSAGE_MAP()
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
};