#pragma once

#include "tstring.h"

// CLogDialog 대화 상자

class CLogDialog : public CDialogEx
{
	DECLARE_DYNAMIC(CLogDialog)
	DECLARE_MESSAGE_MAP()

public:
	CLogDialog(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CLogDialog();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_LOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

public:
	void printLog(const std::tstring &message);
	void initLog();

private:
	CListBox __ddx_list_log;

public:
	afx_msg void OnBnClickedLogButtonDump();
	afx_msg void OnBnClickedLogButtoninitlog();
};
