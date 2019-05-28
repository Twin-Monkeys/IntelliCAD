#include "CSliceView.h"
#include "System.h"

IMPLEMENT_DYNCREATE(CSliceView, CView)

void CSliceView::_onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight)
{
	System::getSystemContents().
		getRenderingEngine().imageProcessor.render(pDevScreen, screenWidth, screenHeight, sliceAxis);
}BEGIN_MESSAGE_MAP(CSliceView, CRenderingView)
ON_WM_MOUSEWHEEL()
END_MESSAGE_MAP()


BOOL CSliceView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
	RenderingEngine& renderingEngine = RenderingEngine::getInstance();

	if (zDelta > 0)
		renderingEngine.volumeRenderer.adjustImgBasedSamplingStep(-0.1f);
	else
		renderingEngine.volumeRenderer.adjustImgBasedSamplingStep(0.1f);

	render();

	return CRenderingView::OnMouseWheel(nFlags, zDelta, pt);
}
