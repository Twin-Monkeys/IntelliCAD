#include "CRenderingView.h"
#include "CCustomSplitterWnd.h"
#include "System.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CRenderingView, CView)
ON_WM_SIZE()
ON_WM_CREATE()
ON_WM_DESTROY()
ON_WM_ERASEBKGND()
ON_WM_LBUTTONDBLCLK()
END_MESSAGE_MAP()

CRenderingView::CRenderingView() :
	_volumeLoaded(__volumeLoaded)
{}

/* member function */
void CRenderingView::OnDraw(CDC* /*pDC*/)
{
	wglMakeCurrent(__hDeviceContext, __hRenderingContext);

	if (__volumeLoaded)
	{
		__onDeviceDraw();

		wglMakeCurrent(__hDeviceContext, nullptr);
		__onHostDraw();

		wglMakeCurrent(__hDeviceContext, __hRenderingContext);
	}
	else
		SwapBuffers(__hDeviceContext);
}

void CRenderingView::__onDeviceDraw()
{
	glDrawPixels(__screenSize.cx, __screenSize.cy, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	SwapBuffers(__hDeviceContext);
}

void CRenderingView::__onHostDraw()
{
	CDC *const pDCInterface = CDC::FromHandle(__hDeviceContext);
	_onHostRender(pDCInterface, __screenSize.cx, __screenSize.cy);
}

void CRenderingView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	wglMakeCurrent(__hDeviceContext, __hRenderingContext);

	__deleteBuffer();
	__createBuffer(cx, cy);
	_render();
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

	/*
		DC�� ���� �׷����� ����(�簢��)�� ���Եǳ�,
		�ȼ��� ���ŵ��� �ʴ� ������ ó���ϴ� ����� �����Ѵ�. (�������� ó��)
		�� �۾��� hostRender() �Լ��� ���� �۾���.
	*/
	SetBkMode(__hDeviceContext, TRANSPARENT);

	// pixelDesc�� �뺯�ϴ� pixel format�� ��ȣ�� ��� �´�.
	int format = ChoosePixelFormat(__hDeviceContext, &pixelDesc);
	SetPixelFormat(__hDeviceContext, format, &pixelDesc);

	// openGL rendering context�� �����. 
	__hRenderingContext = wglCreateContext(__hDeviceContext);

	wglMakeCurrent(__hDeviceContext, __hRenderingContext);
	glewInit();

	// ���� ����ȭ(����� �ִ� �ֻ����� ������ ��� ������ ����)�� �����Ѵ�.
	wglSwapIntervalEXT(0);

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();

	eventBroadcaster.addVolumeLoadedListener(*this);
	eventBroadcaster.addRequestScreenUpdateListener(*this);

	return 0;
}

void CRenderingView::OnDestroy()
{
	CView::OnDestroy();

	__deleteBuffer();

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

void CRenderingView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	CCustomSplitterWnd* pSplitterWnd = (CCustomSplitterWnd*)GetParentSplitter(this, false);
	pSplitterWnd->maximized = !(pSplitterWnd->maximized);

	if (pSplitterWnd->maximized) 
		pSplitterWnd->maximizeActiveView(index);
	else
		pSplitterWnd->updateView();

	CView::OnLButtonDblClk(nFlags, point);
}

void CRenderingView::_render()
{
	// ������ �ֵ����� CUDA�� �����´�.
	cudaGraphicsMapResources(1, &__pCudaRes, nullptr);

	// �ڵ��� ����Ű�� ������ ���� �����͸� �����´�.
	Pixel* pDevScreen = nullptr;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pDevScreen), nullptr, __pCudaRes);

	if (pDevScreen)
		_onDeviceRender(pDevScreen, __screenSize.cx, __screenSize.cy);

	// ������ �ֵ����� GL�� �����´�.
	cudaGraphicsUnmapResources(1, &__pCudaRes, nullptr);

	// ȭ�� ������ ��û�Ѵ�.
	Invalidate();
}

void CRenderingView::_onHostRender(CDC *const pDC, const int screenWidth, const int screenHeight)
{}

const CSize &CRenderingView::_getScreenSize() const
{
	return __screenSize;
}

void CRenderingView::__createBuffer(const int width, const int height)
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

void CRenderingView::__deleteBuffer()
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

void CRenderingView::init(const int viewIndex)
{
	index = viewIndex;
}

void CRenderingView::onVolumeLoaded(const VolumeMeta &volumeMeta)
{
	__volumeLoaded = true;
	_render();
}

void CRenderingView::onRequestScreenUpdate(const RenderingScreenType targetType)
{
	if (targetType == screenType)
		_render();
}