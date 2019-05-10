#pragma once

#include <afxext.h>

class CCustomSplitterWnd : public CSplitterWnd 
{
	DECLARE_MESSAGE_MAP()

public:
	/* constructor */
	CCustomSplitterWnd();
	CCustomSplitterWnd(const float columnRatio, const float rowRatio);

	/* member function */
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);

	/* member variable */
	bool splitted = false;

private:
	/* member variable */
	CRect __clientWindow;
	float __columnRatio = 0.5f;
	float __rowRatio = 0.5f;
};