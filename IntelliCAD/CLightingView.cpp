#include "stdafx.h"
#include "IntelliCAD.h"
#include "CLightingView.h"

IMPLEMENT_DYNCREATE(CLightingView, CFormView)

CLightingView::CLightingView() :
	CFormView(IDD_FORM_VIEW_LIGHTING)
{
}

CLightingView::~CLightingView()
{
}

void CLightingView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}
