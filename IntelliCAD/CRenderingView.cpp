#include "stdafx.h"
#include "CRenderingView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CRenderingView, CView)
ON_WM_SIZE()
ON_WM_CREATE()
ON_WM_DESTROY()
ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

/* member function */
void CRenderingView::OnDraw(CDC* /*pDC*/)
{
	wglMakeCurrent(__hDeviceContext, __hRenderingContext);
	glDrawPixels(__screenSize.cx, __screenSize.cy, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	SwapBuffers(__hDeviceContext);
}

void CRenderingView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	wglMakeCurrent(__hDeviceContext, __hRenderingContext);

	__deleteDeviceBuffer();
	__createDeviceBuffer(cx, cy);
	__render();
}

int CRenderingView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// openGL 초기화 시작
	PIXELFORMATDESCRIPTOR pixelDesc;
	::ZeroMemory(&pixelDesc, sizeof(pixelDesc));

	pixelDesc.nSize = sizeof(pixelDesc);
	pixelDesc.nVersion = 1;
	pixelDesc.dwFlags = (PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL);
	pixelDesc.iPixelType = PFD_TYPE_RGBA;
	pixelDesc.cColorBits = 24;
	pixelDesc.iLayerType = PFD_MAIN_PLANE;

	// 렌더링 용으로 쓸 device context를 만든다.
	__hDeviceContext = ::GetDC(GetSafeHwnd());

	// pixelDesc가 대변하는 pixel format의 번호를 얻어 온다.
	int format = ChoosePixelFormat(__hDeviceContext, &pixelDesc);
	SetPixelFormat(__hDeviceContext, format, &pixelDesc);

	// openGL rendering context를 만든다. 
	__hRenderingContext = wglCreateContext(__hDeviceContext);

	wglMakeCurrent(__hDeviceContext, __hRenderingContext);
	glewInit();

	// 수직 동기화(모니터 최대 주사율에 프레임 계산 성능을 제한)를 해제한다.
	wglSwapIntervalEXT(0);

	return 0;
}

void CRenderingView::OnDestroy()
{
	CView::OnDestroy();

	__deleteDeviceBuffer();

	// device context 제어권을 돌려 받는다.
	wglMakeCurrent(__hDeviceContext, nullptr);

	// openGL rendering context를 제거한다.
	wglDeleteContext(__hRenderingContext);

	// device context를 제거한다.
	::ReleaseDC(GetSafeHwnd(), __hDeviceContext);
}

BOOL CRenderingView::OnEraseBkgnd(CDC* pDC)
{
	return true;
	// return CView::OnEraseBkgnd(pDC);
}

void CRenderingView::__createDeviceBuffer(const int width, const int height)
{
	__screenSize.cx = width;
	__screenSize.cy = height;

	// 버퍼를 만든다.
	glGenBuffers(1, &__bufferObject);

	// 버퍼를 렌더링 아웃풋으로 설정한다.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, __bufferObject);

	// 버퍼 정보를 준다.
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (width * height * sizeof(Pixel)), nullptr, GL_DYNAMIC_DRAW_ARB);

	// 하나의 버퍼를 CUDA와 GL이 나눠 쓴다.
	cudaGraphicsGLRegisterBuffer(&__pCudaRes, __bufferObject, cudaGraphicsMapFlags::cudaGraphicsMapFlagsWriteDiscard);
}

void CRenderingView::__deleteDeviceBuffer()
{
	if (__pCudaRes)
	{
		// 버퍼 공유를 해제한다.
		cudaGraphicsUnregisterResource(__pCudaRes);
		__pCudaRes = nullptr;

		// 렌더링 아웃풋을 프레임 버퍼(기본)로 설정한다.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// 버퍼를 제거한다.
		glDeleteBuffers(1, &__bufferObject);
		__bufferObject = 0;
	}
}

void CRenderingView::__render()
{
	// 버퍼의 주도권을 CUDA로 가져온다.
	cudaGraphicsMapResources(1, &__pCudaRes, nullptr);

	// 핸들이 가리키는 버퍼의 시작 포인터를 가져온다.
	Pixel* pDevScreen = nullptr;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pDevScreen), nullptr, __pCudaRes);

	if (pDevScreen)
		_onRender(pDevScreen, __screenSize.cx, __screenSize.cy);

	// 버퍼의 주도권을 GL로 가져온다.
	cudaGraphicsUnmapResources(1, &__pCudaRes, nullptr);

	// 화면 갱신을 요청한다.
	Invalidate();
}