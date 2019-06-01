#pragma once

#include "CRenderingView.h"
#include "SliceAxis.h"

class CSliceView : public CRenderingView 
{
	DECLARE_DYNCREATE(CSliceView)
	DECLARE_MESSAGE_MAP()

protected:
	/* member function */
	virtual void _onDeviceRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;
	virtual void _onHostRender(CDC *const pDC, const int screenWidth, const int screenHeight) override;

public:
	SliceAxis sliceAxis = SliceAxis::TOP;

private:
	float __samplingStep = 1.f;

	bool __mButtonDown = false;
	CPoint __prevPos;

	CPen __dcPen;
	CBrush __dcBrush;

public:
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMButtonUp(UINT nFlags, CPoint point);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
};