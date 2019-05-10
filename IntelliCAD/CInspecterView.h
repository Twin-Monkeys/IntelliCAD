#pragma once
#include <afxext.h>

class CInspecterView : public CFormView
{
	DECLARE_DYNCREATE(CInspecterView)

public:
	CInspecterView();
	virtual ~CInspecterView();

#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG1 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);
};

