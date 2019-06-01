#include "CVolumeRenderingView.h"
#include "System.h"
#include "Constant.h"

IMPLEMENT_DYNCREATE(CVolumeRenderingView, CView)

BEGIN_MESSAGE_MAP(CVolumeRenderingView, CRenderingView)
	ON_WM_LBUTTONDOWN()
	ON_WM_RBUTTONDOWN()
	ON_WM_MBUTTONDOWN()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_LBUTTONUP()
	ON_WM_MBUTTONUP()
	ON_WM_RBUTTONUP()
END_MESSAGE_MAP()

/* member function */
void CVolumeRenderingView::OnLButtonDown(UINT nFlags, CPoint point)
{
	__lButtonDown = true;
	__prevPos = point;

	CRenderingView::OnLButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnRButtonDown(UINT nFlags, CPoint point)
{
	__rButtonDown = true;
	__prevPos = point;

	CRenderingView::OnRButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnMButtonDown(UINT nFlags, CPoint point)
{
	__mButtonDown = true;
	__prevPos = point;

	CRenderingView::OnMButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnMouseMove(UINT nFlags, CPoint point)
{
	if (!(nFlags & MK_LBUTTON))
		__lButtonDown = false;

	if (!(nFlags & MK_MBUTTON))
		__mButtonDown = false;

	if (!(nFlags & MK_RBUTTON))
		__rButtonDown = false;

	if (__lButtonDown || __mButtonDown || __rButtonDown)
	{
		Camera& camera =
			System::getSystemContents().getRenderingEngine().volumeRenderer.camera;

		if (__lButtonDown)
		{
			if (__dblClickSemaphore)
				__dblClickSemaphore = false;
			else
			{
				const float DELTA_X = (static_cast<float>(point.x - __prevPos.x) * 0.005f);
				const float DELTA_Y = (static_cast<float>(point.y - __prevPos.y) * 0.005f);

				camera.orbitYaw(Constant::Volume::PIVOT, -DELTA_X);
				camera.orbitPitch(Constant::Volume::PIVOT, DELTA_Y);

				__prevPos.x = point.x;
				__prevPos.y = point.y;
			}
		}

		if (__mButtonDown)
		{
			const float DELTA_X = static_cast<float>(point.x - __prevPos.x);
			const float DELTA_Y = static_cast<float>(point.y - __prevPos.y);

			camera.adjustHorizontal(-DELTA_X);
			camera.adjustVertical(-DELTA_Y);

			__prevPos.x = point.x;
			__prevPos.y = point.y;
		}

		if (__rButtonDown)
		{
			const float DELTA_Y = (static_cast<float>(point.y - __prevPos.y) * 0.005f);

			camera.orbitRoll(Constant::Volume::PIVOT, DELTA_Y);

			__prevPos.y = point.y;
		}

		render();
	}

	CRenderingView::OnMouseMove(nFlags, point);
}

void CVolumeRenderingView::_onDeviceRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight)
{
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.render(pDevScreen, screenWidth, screenHeight);
}

BOOL CVolumeRenderingView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	RenderingEngine::VolumeRenderer& volumeRenderer =
		System::getSystemContents().getRenderingEngine().volumeRenderer;

	if (zDelta > 0)
		volumeRenderer.adjustImgBasedSamplingStep(-0.1f);
	else
		volumeRenderer.adjustImgBasedSamplingStep(0.1f);

	render();

	return CRenderingView::OnMouseWheel(nFlags, zDelta, pt);
}


void CVolumeRenderingView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	__dblClickSemaphore = true;

	CRenderingView::OnLButtonDblClk(nFlags, point);
}


void CVolumeRenderingView::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	__lButtonDown = false;

	CRenderingView::OnLButtonUp(nFlags, point);
}


void CVolumeRenderingView::OnMButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	__mButtonDown = false;

	CRenderingView::OnMButtonUp(nFlags, point);
}


void CVolumeRenderingView::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	__rButtonDown = false;

	CRenderingView::OnRButtonUp(nFlags, point);
}
