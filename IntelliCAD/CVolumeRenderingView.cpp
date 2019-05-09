#include "CVolumeRenderingView.h"
#include "System.h"

IMPLEMENT_DYNCREATE(CVolumeRenderingView, CView)

BEGIN_MESSAGE_MAP(CVolumeRenderingView, CRenderingView)
END_MESSAGE_MAP()

void CVolumeRenderingView::_onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight)
{
	System::getSystemContents().
		getRenderingEngine().render(pDevScreen, screenWidth, screenHeight);
}