// CImageInfoDialog.cpp: 구현 파일
//

#include "stdafx.h"
#include "IntelliCAD.h"
#include "CImageInfoDialog.h"
#include "afxdialogex.h"
#include "System.h"
#include "Parser.hpp"

// CImageInfoDialog 대화 상자

IMPLEMENT_DYNAMIC(CImageInfoDialog, CDialogEx)

BEGIN_MESSAGE_MAP(CImageInfoDialog, CDialogEx)
	ON_WM_CREATE()
END_MESSAGE_MAP()

CImageInfoDialog::CImageInfoDialog(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_IMAGE_INFO, pParent)
{

}

CImageInfoDialog::~CImageInfoDialog()
{
}

void CImageInfoDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_IMG_INFO_EDIT_imageName, __ddx_imageName);

	DDX_Control(pDX, IDC_IMG_INFO_EDIT_memSizeW, __ddx_memSizeW);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_memSizeH, __ddx_memSizeH);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_memSizeD, __ddx_memSizeD);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_memSizeT, __ddx_memSizeT);

	DDX_Control(pDX, IDC_IMG_INFO_EDIT_spacingX, __ddx_spacingX);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_spacingY, __ddx_spacingY);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_spacingZ, __ddx_spacingZ);

	DDX_Control(pDX, IDC_IMG_INFO_EDIT_volSizeW, __ddx_volSizeW);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_volSizeH, __ddx_volSizeH);
	DDX_Control(pDX, IDC_IMG_INFO_EDIT_volSizeD, __ddx_volSizeD);
}


// CImageInfoDialog 메시지 처리기

void CImageInfoDialog::onVolumeLoaded(const VolumeMeta &volumeMeta)
{
	const VolumeNumericMeta &volumeNumericMeta = volumeMeta.numeric;

	CString tmp;

	// Image name
	tmp = Parser::tstring$CString(volumeMeta.fileName);
	__ddx_imageName.SetWindowText(tmp);

	// Mem size - width
	tmp.Format(_T("%d"), volumeNumericMeta.memSize.width);
	__ddx_memSizeW.SetWindowText(tmp);

	// Mem size - height
	tmp.Format(_T("%d"), volumeNumericMeta.memSize.height);
	__ddx_memSizeH.SetWindowText(tmp);

	// Mem size - depth
	tmp.Format(_T("%d"), volumeNumericMeta.memSize.depth);
	__ddx_memSizeD.SetWindowText(tmp);

	// Mem size - total
	tmp.Format(_T("%d"), volumeNumericMeta.memSize.getTotalSize());
	__ddx_memSizeT.SetWindowText(tmp);

	// Spacing - x
	tmp.Format(_T("%f"), volumeNumericMeta.spacing.width);
	__ddx_spacingX.SetWindowText(tmp);

	// Spacing - y
	tmp.Format(_T("%f"), volumeNumericMeta.spacing.height);
	__ddx_spacingY.SetWindowText(tmp);

	// Spacing - z
	tmp.Format(_T("%f"), volumeNumericMeta.spacing.depth);
	__ddx_spacingZ.SetWindowText(tmp);

	// Vol size - width
	tmp.Format(_T("%f"), volumeNumericMeta.volSize.width);
	__ddx_volSizeW.SetWindowText(tmp);

	// Vol size - height
	tmp.Format(_T("%f"), volumeNumericMeta.volSize.height);
	__ddx_volSizeH.SetWindowText(tmp);

	// Vol size - depth
	tmp.Format(_T("%f"), volumeNumericMeta.volSize.depth);
	__ddx_volSizeD.SetWindowText(tmp);
}

int CImageInfoDialog::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.
	System::getSystemContents().getEventBroadcaster().addVolumeLoadedListener(*this);

	return 0;
}

void CImageInfoDialog::OnOK()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	// __super::OnOK();
}


void CImageInfoDialog::OnCancel()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	// __super::OnCancel();
}
