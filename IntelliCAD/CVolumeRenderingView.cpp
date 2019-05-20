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
END_MESSAGE_MAP()

/* member function */
void CVolumeRenderingView::OnLButtonDown(UINT nFlags, CPoint point)
{
	__prevPos.x = point.x;
	__prevPos.y = point.y;

	CRenderingView::OnLButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnRButtonDown(UINT nFlags, CPoint point)
{
	__prevPos.y = point.y;

	CRenderingView::OnRButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnMButtonDown(UINT nFlags, CPoint point)
{
	__prevPos.x = point.x;
	__prevPos.y = point.y;

	CRenderingView::OnMButtonDown(nFlags, point);
}

void CVolumeRenderingView::OnMouseMove(UINT nFlags, CPoint point)
{
	if (nFlags == MK_LBUTTON || nFlags == MK_MBUTTON || nFlags == MK_RBUTTON)
	{
		Camera& camera = RenderingEngine::getInstance().volumeRenderer.camera;

		switch (nFlags)
		{
		case MK_LBUTTON:
		{
			const float DELTA_X = (static_cast<float>(point.x - __prevPos.x) * 0.007f);
			const float DELTA_Y = (static_cast<float>(point.y - __prevPos.y) * 0.007f);

			camera.orbitYaw(Constant::Volume::PIVOT, -DELTA_X);
			camera.orbitPitch(Constant::Volume::PIVOT, DELTA_Y);

			__prevPos.x = point.x;
			__prevPos.y = point.y;
		}
		break;

		case MK_MBUTTON:
		{
			const float DELTA_X = static_cast<float>(point.x - __prevPos.x);
			const float DELTA_Y = static_cast<float>(point.y - __prevPos.y);

			camera.adjustHorizontal(-DELTA_X);
			camera.adjustVertical(DELTA_Y);

			__prevPos.x = point.x;
			__prevPos.y = point.y;
		}
		break;

		case MK_RBUTTON:
		{
			const float DELTA_Y = (static_cast<float>(point.y - __prevPos.y) * 0.007f);

			camera.orbitRoll(Constant::Volume::PIVOT, DELTA_Y);

			__prevPos.y = point.y;
		}
		break;
		}

		render();
	}

	CRenderingView::OnMouseMove(nFlags, point);
}

void CVolumeRenderingView::_onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight)
{
	System::getSystemContents().
		getRenderingEngine().volumeRenderer.render(pDevScreen, screenWidth, screenHeight);
}

BOOL CVolumeRenderingView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	RenderingEngine& renderingEngine = RenderingEngine::getInstance();

	if (zDelta > 0)
		renderingEngine.volumeRenderer.adjustImgBasedSamplingStep(-0.1f);
	else
		renderingEngine.volumeRenderer.adjustImgBasedSamplingStep(0.1f);

	render();

	return CRenderingView::OnMouseWheel(nFlags, zDelta, pt);
}
