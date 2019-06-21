#pragma once

#include "tstring.h"

// CServerInteractionActiveDlg 대화 상자

class CServerInteractionActiveDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CServerInteractionActiveDlg)
	DECLARE_MESSAGE_MAP()

public:
	CServerInteractionActiveDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CServerInteractionActiveDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_SERVER_INTERACTION_ACTIVE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

private:
	CStatic __ddx_userId;

public:
	void setUserId(const std::tstring &id);
	afx_msg void OnBnClickedServerActiveButtonImgDownTest();
};
