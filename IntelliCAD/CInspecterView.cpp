#include "stdafx.h"
#include "IntelliCAD.h"
#include "CInspecterView.h"

IMPLEMENT_DYNCREATE(CInspecterView, CFormView)

CInspecterView::CInspecterView() : 
	CFormView(IDD_DIALOG1)
{
}

CInspecterView::~CInspecterView()
{
}

void CInspecterView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}

