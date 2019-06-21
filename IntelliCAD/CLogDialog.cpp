// CLogDialog.cpp: 구현 파일
//

#include "stdafx.h"
#include <ctime>
#include <chrono>
#include "CLogDialog.h"
#include "IntelliCAD.h"
#include "afxdialogex.h"
#include "Parser.hpp"
#include "System.h"
#include "tfstream.h"

using namespace std;


// CLogDialog 대화 상자

IMPLEMENT_DYNAMIC(CLogDialog, CDialogEx)

BEGIN_MESSAGE_MAP(CLogDialog, CDialogEx)
	ON_BN_CLICKED(IDC_LOG_BUTTON_dump, &CLogDialog::OnBnClickedLogButtonDump)
	ON_BN_CLICKED(IDC_LOG_BUTTON_initLog, &CLogDialog::OnBnClickedLogButtoninitlog)
END_MESSAGE_MAP()

CLogDialog::CLogDialog(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_LOG, pParent)
{
	System::getInstance().__setLogDlgReference(*this);
}

CLogDialog::~CLogDialog()
{
}

void CLogDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LOG_LISTBOX_log, __ddx_list_log);
}


// CLogDialog 메시지 처리기

static std::tstring __buildCurrentTimeString(const bool spaceAndColon = true)
{
	using namespace chrono;

	auto now = system_clock::now();
	time_t now_t = system_clock::to_time_t(now);

	tm now_tm;
	localtime_s(&now_tm, &now_t);

	char buffer[32];

	if (spaceAndColon)
		strftime(buffer, sizeof(buffer), "[%H:%M:%S] ", &now_tm);
	else
		strftime(buffer, sizeof(buffer), "[%H_%M_%S]", &now_tm);

	return Parser::LPCSTR$tstring(buffer);
}

void CLogDialog::printLog(const tstring &message)
{
	const tstring LOG_STRING = (__buildCurrentTimeString() + message);

	__ddx_list_log.AddString(LOG_STRING.c_str());
	__ddx_list_log.SetTopIndex(__ddx_list_log.GetCount() - 1);
}

void CLogDialog::initLog()
{
	__ddx_list_log.ResetContent();
}

void CLogDialog::OnBnClickedLogButtonDump()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	const tstring CUR_TIME_STRING = __buildCurrentTimeString(false);

	const TCHAR *const szFilter = _T("IntelliCAD Log Dump (*.txt) |*.txt|");
	CFileDialog fileDlg(
		false, _T("IntelliCAD Log Dump (*.txt)"),
		(_T("IntelliCAD Log Dump ") + CUR_TIME_STRING).c_str(), 6UL, szFilter);

	if (fileDlg.DoModal() == IDOK)
	{
		tofstream fout;

		do
		{
			fout.open(fileDlg.GetPathName());

			if (!fout)
			{
				const int MB_RESULT = MessageBox(
					_T("해당 경로에 로그 파일을 덤프할 수 없습니다."),
					_T("로그 덤프 오류"), MB_ICONEXCLAMATION | MB_RETRYCANCEL);

				if (MB_RESULT == IDRETRY)
					continue;

				return;
			}

			break;
		}
		while (true);

		const int ITER = __ddx_list_log.GetCount();
		for (int i = 0; i < ITER; i++)
		{
			CString tmp;
			__ddx_list_log.GetText(i, tmp);

			fout << tmp.GetString() << endl;
		}

		MessageBox(_T("로그 덤프가 완료되었습니다."), _T("로그 덤프 완료"), MB_ICONINFORMATION);
	}
}

void CLogDialog::OnBnClickedLogButtoninitlog()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	const int MB_RESULT =
		MessageBox(_T("로그를 초기화 하시겠습니까?"), _T("로그 초기화"), MB_ICONQUESTION | MB_YESNO);

	if (MB_RESULT == IDYES)
		initLog();
}
