#pragma once

#include <afxext.h>

class CCustomSplitterWnd : public CSplitterWnd
{
	DECLARE_MESSAGE_MAP()

public:
	/* constructor */
	CCustomSplitterWnd(class CMainFrame& mainFrm);

	/* member function */
	afx_msg void OnSize(const UINT nType, const int cx, const int cy);

private:
	CMainFrame& __mainFrm;
};
