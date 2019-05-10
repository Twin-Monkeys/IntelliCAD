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
	// ���� ������ ���� ������ �����Ѵ�.
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
	// Splitter Bar�� ��ġ ������ �����´�.
	CSplitterWnd::OnLButtonUp(nFlags, point);
	
	const float WIDTH = static_cast<float>(m_pColInfo->nIdealSize);
	const float HEIGHT = static_cast<float>(m_pRowInfo->nIdealSize);

	// Ŭ���̾�Ʈ ������ ũ�� ������ �����´�. 
	const float WINDOW_WIDTH = static_cast<float>(__clientWindow.Width());
	const float WINDOW_HEIGHT = static_cast<float>(__clientWindow.Height());

	// Splitter Bar�� ���� ������ ����Ѵ�.
	__columnRatio = (WIDTH / WINDOW_WIDTH);
	__rowRatio = (HEIGHT / WINDOW_HEIGHT);
}
