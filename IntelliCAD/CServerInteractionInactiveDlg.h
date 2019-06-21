#pragma once


// CServerInteractionInactiveDlg 대화 상자

class CServerInteractionInactiveDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CServerInteractionInactiveDlg)
	DECLARE_MESSAGE_MAP()

public:
	CServerInteractionInactiveDlg(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~CServerInteractionInactiveDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_SERVER_INTERACTION_INACTIVE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.
};
