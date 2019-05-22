#pragma once

#define SIGN_IN 519

// CLoginDlg 대화 상자
class CLoginDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CLoginDlg)
	DECLARE_MESSAGE_MAP()

public:
	CLoginDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CLoginDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG2 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

public:
	afx_msg void OnBnClickedButton1();

private:
	void __loadLogo();
	CImage __logo;
public:
	afx_msg void OnPaint();
	virtual BOOL PreTranslateMessage(MSG* pMsg);
};
