// CCustomRibbonBar.cpp: 구현 파일
//

#include "CCustomRibbonBar.h"


// CCustomRibbonBar

IMPLEMENT_DYNAMIC(CCustomRibbonBar, CMFCRibbonBar)

CCustomRibbonBar::CCustomRibbonBar()
{
}

CCustomRibbonBar::~CCustomRibbonBar()
{
}


BEGIN_MESSAGE_MAP(CCustomRibbonBar, CMFCRibbonBar)
END_MESSAGE_MAP()



// CCustomRibbonBar 메시지 처리기

void CCustomRibbonBar::DeleteDropdown()
{
	// TODO: 여기에 구현 코드 추가.
	m_QAToolbar.RemoveAll();
}