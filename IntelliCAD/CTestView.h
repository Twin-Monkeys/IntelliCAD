#pragma once

#include "CRenderingView.h"

class CTestView : public CRenderingView 
{
	DECLARE_DYNCREATE(CTestView)

protected:
	/* member function */
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;
};