#pragma once
#include <afxext.h>

class CLightingView : public CFormView
{
	DECLARE_DYNCREATE(CLightingView)

public:
	CLightingView();
	virtual ~CLightingView();

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_FORM_VIEW_LIGHTING };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);
};

