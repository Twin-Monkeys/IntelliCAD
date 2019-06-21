#pragma once

#define LOGIN_DLG_LOGIN_SUCCESS 519

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
	enum { IDD = IDD_DIALOG_LOGIN };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

public:
	afx_msg void OnBnClickedButtonSignIn();

private:
	CEdit __ddxc_id;
	CEdit __ddxc_pw;

public:
	afx_msg void OnBnClickedButtonClose();
	virtual void OnOK();
	virtual void OnCancel();
};
