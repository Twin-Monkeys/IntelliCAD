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

	// openGL �ʱ�ȭ ����
	PIXELFORMATDESCRIPTOR pixelDesc;
	::ZeroMemory(&pixelDesc, sizeof(pixelDesc));

	pixelDesc.nSize = sizeof(pixelDesc);
	pixelDesc.nVersion = 1;
	pixelDesc.dwFlags = (PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL);
	pixelDesc.iPixelType = PFD_TYPE_RGBA;
	pixelDesc.cColorBits = 24;
	pixelDesc.iLayerType = PFD_MAIN_PLANE;

	// ������ ������ �� device context�� �����.
	__hDeviceContext = ::GetDC(GetSafeHwnd());

	// pixelDesc�� �뺯�ϴ� pixel format�� ��ȣ�� ��� �´�.
	int format = ChoosePixelFormat(__hDeviceContext, &pixelDesc);
	SetPixelFormat(__hDeviceContext, format, &pixelDesc);

	// openGL rendering context�� �����. 
	__hRenderingContext = wglCreateContext(__hDeviceContext);

	wglMakeCurrent(__hDeviceContext, __hRenderingContext);
	glewInit();

	// ���� ����ȭ(����� �ִ� �ֻ����� ������ ��� ������ ����)�� �����Ѵ�.
	wglSwapIntervalEXT(0);

	return 0;
}

void CRenderingView::OnDestroy()
{
	CView::OnDestroy();

	__deleteDeviceBuffer();

	// device context ������� ���� �޴´�.
	wglMakeCurrent(__hDeviceContext, nullptr);

	// openGL rendering context�� �����Ѵ�.
	wglDeleteContext(__hRenderingContext);

	// device context�� �����Ѵ�.
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

	// ���۸� �����.
	glGenBuffers(1, &__bufferObject);

	// ���۸� ������ �ƿ�ǲ���� �����Ѵ�.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, __bufferObject);

	// ���� ������ �ش�.
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (width * height * sizeof(Pixel)), nullptr, GL_DYNAMIC_DRAW_ARB);

	// �ϳ��� ���۸� CUDA�� GL�� ���� ����.
	cudaGraphicsGLRegisterBuffer(&__pCudaRes, __bufferObject, cudaGraphicsMapFlags::cudaGraphicsMapFlagsWriteDiscard);
}

void CRenderingView::__deleteDeviceBuffer()
{
	if (__pCudaRes)
	{
		// ���� ������ �����Ѵ�.
		cudaGraphicsUnregisterResource(__pCudaRes);
		__pCudaRes = nullptr;

		// ������ �ƿ�ǲ�� ������ ����(�⺻)�� �����Ѵ�.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		// ���۸� �����Ѵ�.
		glDeleteBuffers(1, &__bufferObject);
		__bufferObject = 0;
	}
}

void CRenderingView::__render()
{
	// ������ �ֵ����� CUDA�� �����´�.
	cudaGraphicsMapResources(1, &__pCudaRes, nullptr);

	// �ڵ��� ����Ű�� ������ ���� �����͸� �����´�.
	Pixel* pDevScreen = nullptr;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pDevScreen), nullptr, __pCudaRes);

	if (pDevScreen)
		_onRender(pDevScreen, __screenSize.cx, __screenSize.cy);

	// ������ �ֵ����� GL�� �����´�.
	cudaGraphicsUnmapResources(1, &__pCudaRes, nullptr);

	// ȭ�� ������ ��û�Ѵ�.
	Invalidate();
}