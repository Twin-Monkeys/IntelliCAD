#pragma once

#include "stdafx.h"
#include <GL/glew.h>
#include <GL/wglew.h>
#include <cuda_gl_interop.h>
#include "Pixel.h"
#include "VolumeLoadingListener.h"

class CRenderingView : public CView, public VolumeLoadingListener
{
	DECLARE_MESSAGE_MAP()

public:
	/* member function */
	virtual void OnDraw(CDC* /*pDC*/);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	afx_msg void OnLButtonDblClk(UINT nFlags, CPoint point);
	void render();

	virtual void onLoadVolume(const VolumeData &volumeData) override;

	/* member variable */
	int index;

protected:
	/* member function */
	virtual void _onDeviceRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) = 0;
	virtual void _onHostRender(CDC *const pDC, const int screenWidth, const int screenHeight);

	const CSize &_getScreenSize() const;

private:
	/* member function */
	void __createBuffer(const int width, const int height);
	void __deleteBuffer();

	void __onDeviceDraw();
	void __onHostDraw();

	/* member variable */
	HDC __hDeviceContext = nullptr;
	HGLRC __hRenderingContext = nullptr;
	CSize __screenSize;
	GLuint __bufferObject = 0;
	cudaGraphicsResource* __pCudaRes = nullptr;

	bool __initialized = false;
};