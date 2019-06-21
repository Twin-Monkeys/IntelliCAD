#include "CSliceView.h"
#include "System.h"
#include "MainFrm.h"

IMPLEMENT_DYNCREATE(CSliceView, CView)

BEGIN_MESSAGE_MAP(CSliceView, CRenderingView)
	ON_WM_MOUSEWHEEL()
	ON_WM_MOUSEMOVE()
	ON_WM_MBUTTONDOWN()
	ON_WM_MBUTTONUP()
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONDBLCLK()
END_MESSAGE_MAP()

CSliceView::CSliceView()
{
	screenType = RenderingScreenType::SLICE_TOP;
}

void CSliceView::_onDeviceRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight)
{
	System::getSystemContents().getRenderingEngine().
		imageProcessor.render(pDevScreen, screenWidth, screenHeight, sliceAxis);
}

void CSliceView::_onHostRender(CDC *const pDC, const int screenWidth, const int screenHeight)
{
	Index2D<> slicingPoint = System::getSystemContents().getRenderingEngine().
		imageProcessor.getSlicingPointForScreen({ screenWidth, screenHeight }, sliceAxis);

	CPen *pPrevPen = pDC->SelectObject(&__dcPen);
	CBrush *pPrevBrush = pDC->SelectObject(&__dcBrush);

	pDC->MoveTo(0, slicingPoint.y);
	pDC->LineTo(screenWidth, slicingPoint.y);

	pDC->MoveTo(slicingPoint.x, 0);
	pDC->LineTo(slicingPoint.x, screenHeight);

	pDC->Ellipse(
		slicingPoint.x - 5,
		slicingPoint.y - 5,
		slicingPoint.x + 5,
		slicingPoint.y + 5
	);

	pDC->SelectObject(pPrevPen);
	pDC->SelectObject(pPrevBrush);
}

void CSliceView::__onMButtonUpImpl()
{
	__mButtonDown = false;
	System::getSystemContents().
		getEventBroadcaster().notifyUpdateAnchorFromView(sliceAxis);
}

void CSliceView::init(const int viewIndex, const SliceAxis sliceAxis)
{
	__super::init(viewIndex);

	this->sliceAxis = sliceAxis;

	switch (sliceAxis)
	{
	case SliceAxis::TOP:
		screenType = RenderingScreenType::SLICE_TOP;
		break;

	case SliceAxis::FRONT:
		screenType = RenderingScreenType::SLICE_FRONT;
		break;

	case SliceAxis::RIGHT:
		screenType = RenderingScreenType::SLICE_RIGHT;
		break;
	}
}

BOOL CSliceView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!_volumeLoaded)
		return false;

	RenderingEngine::ImageProcessor& imgProcessor =
		System::getSystemContents().getRenderingEngine().imageProcessor;

	if (nFlags & MK_CONTROL)
	{
		if (zDelta > 0)
			imgProcessor.adjustSlicingPoint(-3.f, sliceAxis);
		else
			imgProcessor.adjustSlicingPoint(3.f, sliceAxis);

		EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();
		eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_TOP);
		eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_FRONT);
		eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_RIGHT);
	}
	else
	{
		if (zDelta > 0)
			imgProcessor.adjustSamplingStep(-.1f, sliceAxis);
		else
			imgProcessor.adjustSamplingStep(.1f, sliceAxis);

		_render();
	}

	return CRenderingView::OnMouseWheel(nFlags, zDelta, pt);
}

void CSliceView::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!_volumeLoaded)
		return;

	if (__mButtonDown && !(nFlags & MK_MBUTTON))
		__onMButtonUpImpl();

	if (__mButtonDown)
	{
		RenderingEngine::ImageProcessor& imgProcessor =
			System::getSystemContents().getRenderingEngine().imageProcessor;

		const float DELTA_X = static_cast<float>(point.x - __prevPos.x);
		const float DELTA_Y = static_cast<float>(point.y - __prevPos.y);

		imgProcessor.adjustAnchor(-DELTA_X, DELTA_Y, sliceAxis);

		__prevPos = point;

		_render();
	}

	CRenderingView::OnMouseMove(nFlags, point);
}

void CSliceView::OnMButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!_volumeLoaded)
		return;

	__mButtonDown = true;
	__prevPos = point;

	CRenderingView::OnMButtonDown(nFlags, point);
}

void CSliceView::OnMButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	__onMButtonUpImpl();

	CRenderingView::OnMButtonUp(nFlags, point);
}

int CSliceView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CRenderingView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.
	__dcPen.CreatePen(PS_DOT, 1, RGB(220, 25, 72));
	__dcBrush.CreateSolidBrush(RGB(220, 25, 72));

	System::getSystemContents().
		getEventBroadcaster().addUpdateSliceTransferFunctionListener(*this);

	return 0;
}

void CSliceView::OnDestroy()
{
	CRenderingView::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	__dcPen.DeleteObject();
	__dcBrush.DeleteObject();
}

void CSliceView::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!_volumeLoaded)
		return;

	const CSize &screenSize = _getScreenSize();

	System::getSystemContents().getRenderingEngine().imageProcessor.
		setSlicingPointFromScreen({ screenSize.cx, screenSize.cy }, { point.x, point.y }, sliceAxis);

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_TOP);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_FRONT);
	eventBroadcaster.notifyRequestScreenUpdate(RenderingScreenType::SLICE_RIGHT);
	eventBroadcaster.notifyUpdateSlicingPointFromView();

	CRenderingView::OnLButtonDown(nFlags, point);
}

void CSliceView::onUpdateSliceTransferFunction()
{
	_render();
}

void CSliceView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	if (!_volumeLoaded)
		return;

	__super::OnLButtonDblClk(nFlags, point);
}
