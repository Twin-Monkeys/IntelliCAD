#pragma once

#include "CRenderingView.h"

class CVolumeRenderingView : public CRenderingView
{
	DECLARE_DYNCREATE(CVolumeRenderingView)
	DECLARE_MESSAGE_MAP()

protected:
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;
};

