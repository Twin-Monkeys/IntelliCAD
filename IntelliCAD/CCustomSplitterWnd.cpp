#include "stdafx.h"
#include "CCustomSplitterWnd.h"
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CCustomSplitterWnd, CSplitterWnd)
	ON_WM_SIZE()
	ON_WM_LBUTTONUP()
END_MESSAGE_MAP()

CCustomSplitterWnd::CCustomSplitterWnd() 
{
	// 분할 윈도우 사이 간격을 설정한다.
	m_cxSplitterGap = m_cySplitterGap = 3;
}

void CCustomSplitterWnd::OnSize(const UINT nType, const int cx, const int cy)
{
	CSplitterWnd::OnSize(nType, cx, cy);
	
	if (splitted)
	{
		GetWindowRect(&__clientWindow);
		
		const float WINDOW_WIDTH = static_cast<float>(__clientWindow.Width());
		const float WINDOW_HEIGHT = static_cast<float>(__clientWindow.Height());

		const int COLUMN = static_cast<int>(WINDOW_WIDTH * __columnRatio);
		const int ROW = static_cast<int>(WINDOW_HEIGHT * __rowRatio);

		SetColumnInfo(0, COLUMN, 0);
		SetRowInfo(0, ROW, 0);
		RecalcLayout();
	}
}

void CCustomSplitterWnd::OnLButtonUp(UINT nFlags, CPoint point)
{
	// Splitter Bar의 위치 정보를 가져온다.
	CSplitterWnd::OnLButtonUp(nFlags, point);
	
	const float WIDTH = static_cast<float>(m_pColInfo->nIdealSize);
	const float HEIGHT = static_cast<float>(m_pRowInfo->nIdealSize);

	// 클라이언트 윈도우 크기 정보를 가져온다. 
	const float WINDOW_WIDTH = static_cast<float>(__clientWindow.Width());
	const float WINDOW_HEIGHT = static_cast<float>(__clientWindow.Height());

	// Splitter Bar의 분할 비율을 계산한다.
	__columnRatio = (WIDTH / WINDOW_WIDTH);
	__rowRatio = (HEIGHT / WINDOW_HEIGHT);
}
