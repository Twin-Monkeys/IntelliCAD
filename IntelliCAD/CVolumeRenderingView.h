#pragma once

#include "CRenderingView.h"
#include "UpdateVolumeTransferFunctionListener.h"

class CVolumeRenderingView : public CRenderingView, public UpdateVolumeTransferFunctionListener
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
	virtual void _onDeviceRender(Pixel* const pDevScreen, const int screenWidth, const int screenHeight) override;

private:
	/* member variable */
	bool __lButtonDown = false;
	bool __mButtonDown = false;
	bool __rButtonDown = false;
	CPoint __prevPos;

	bool __dblClickSemaphore = false;

public:
	CVolumeRenderingView();

	afx_msg void OnLButtonDblClk(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);

	virtual void onUpdateVolumeTransferFunction(const ColorChannelType colorType) override;
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
};

