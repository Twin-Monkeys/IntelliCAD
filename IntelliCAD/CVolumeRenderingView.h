#pragma once

#include "CRenderingView.h"

class CVolumeRenderingView : public CRenderingView
{
	DECLARE_DYNCREATE(CVolumeRenderingView)
	DECLARE_MESSAGE_MAP()

public:
	/* member function */
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);

protected:
	/* member function */
	virtual void _onRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;

private:
	/* member variable */
	CPoint __prevPos;
public:
};

