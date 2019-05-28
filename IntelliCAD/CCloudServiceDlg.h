#pragma once


// CCloudServiceDlg 대화 상자

class CCloudServiceDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CCloudServiceDlg)

public:
	CCloudServiceDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CCloudServiceDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_CLOUD_SERVICE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
};
