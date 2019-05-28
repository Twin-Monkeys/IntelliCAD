#pragma once

#include "CRenderingView.h"
#include "SliceAxis.h"

class CSliceView : public CRenderingView 
{
	DECLARE_DYNCREATE(CSliceView)

protected:
	/* member function */
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;

public:
	SliceAxis sliceAxis = SliceAxis::TOP;

private:
	CPoint __prevPos;

};