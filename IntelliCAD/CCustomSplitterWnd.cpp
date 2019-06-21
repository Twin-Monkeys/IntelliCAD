#include "stdafx.h"
#include "CCustomSplitterWnd.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CCustomSplitterWnd, CSplitterWnd)
	ON_WM_SIZE()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDOWN()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()

/* constructor */
CCustomSplitterWnd::CCustomSplitterWnd()
{
	m_cxSplitter = m_cySplitter = 5;
	m_cxBorderShare = m_cyBorderShare = 0;
	m_cxSplitterGap = m_cySplitterGap = 5;
	m_cxBorder = m_cyBorder = 2;
}

CCustomSplitterWnd::CCustomSplitterWnd(const float columnRatio, const float rowRatio) :
	CCustomSplitterWnd()
{
	// Splitter Bar ���� ������ �ʱ�ȭ �Ѵ�.
	__columnRatio = columnRatio;
	__rowRatio = rowRatio;
}

/* member function */
void CCustomSplitterWnd::OnSize(UINT nType, int cx, int cy)
{
	CSplitterWnd::OnSize(nType, cx, cy);

	if (!splitted)
		return;

	// ���� �����찡 ������ ��� Ŭ���̾�Ʈ ������ ũ�� ������ �����´�.
	GetWindowRect(&__clientWindow); 

	if (maximized)
		__maximizeActiveView();
	else
		updateView();
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

void CCustomSplitterWnd::maximizeActiveView(const int viewIdx) 
{
	__maximizedViewIdx = viewIdx;
	__maximizeActiveView();
}

void CCustomSplitterWnd::updateView() 
{
	// Ŭ���̾�Ʈ ������ ũ�� ������ �����´�.
	const float WINDOW_WIDTH = static_cast<float>(__clientWindow.Width());
	const float WINDOW_HEIGHT = static_cast<float>(__clientWindow.Height());

	// Splitter Bar ���� ������ �����Ѵ�.
	const int COLUMN = static_cast<int>(WINDOW_WIDTH * __columnRatio);
	const int ROW = static_cast<int>(WINDOW_HEIGHT * __rowRatio);

	// ���̾ƿ��� �����Ѵ�.
	SetColumnInfo(0, COLUMN, 0);
	SetRowInfo(0, ROW, 0);

	m_cxSplitter = m_cySplitter = 5;
	m_cxSplitterGap = m_cySplitterGap = 5;

	RecalcLayout();
}

void CCustomSplitterWnd::__maximizeActiveView()
{
	switch (__maximizedViewIdx)
	{
	case 0: // Top-Left View�� Ȯ���Ѵ�.
		SetColumnInfo(0, __clientWindow.Width(), 0);
		SetRowInfo(0, __clientWindow.Height(), 0);
		break;

	case 1: // Top-Right View�� Ȯ���Ѵ�.
		SetColumnInfo(0, 0, 0);
		SetRowInfo(0, __clientWindow.Height(), 0);
		break;

	case 2: // Bottom-Left View�� Ȯ���Ѵ�.
		SetColumnInfo(0, __clientWindow.Width(), 0);
		SetRowInfo(0, 0, 0);
		break;

	case 3: // Bottom-Right View�� Ȯ���Ѵ�. 
		SetColumnInfo(0, 0, 0);
		SetRowInfo(0, 0, 0);
		break;
	}

	m_cxSplitter = m_cySplitter = 1;
	m_cxSplitterGap = m_cySplitterGap = 1;

	RecalcLayout();
}

void CCustomSplitterWnd::OnLButtonDown(UINT nFlags, CPoint point)
{
	if (!maximized)
		CSplitterWnd::OnLButtonDown(nFlags, point);
}


void CCustomSplitterWnd::OnMouseMove(UINT nFlags, CPoint point)
{
	if (!maximized)
		CSplitterWnd::OnMouseMove(nFlags, point);
}
