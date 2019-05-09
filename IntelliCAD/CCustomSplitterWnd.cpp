#include "stdafx.h"
#include "CCustomSplitterWnd.h"
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CCustomSplitterWnd, CSplitterWnd)
	ON_WM_SIZE()
END_MESSAGE_MAP()

CCustomSplitterWnd::CCustomSplitterWnd(CMainFrame& mainFrm) :
	__mainFrm(mainFrm)
{}

void CCustomSplitterWnd::OnSize(const UINT nType, const int cx, const int cy)
{
	CSplitterWnd::OnSize(nType, cx, cy);
	
	__mainFrm.notifyClientUpdate();
}
