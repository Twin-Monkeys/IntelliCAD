#pragma once

class CCustomSplitterWnd : public CSplitterWnd 
{
	DECLARE_MESSAGE_MAP()

public:
	/* constructor */
	CCustomSplitterWnd();
	CCustomSplitterWnd(const float columnRatio, const float rowRatio);

	/* member function */
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	void maximizeActiveView(const int viewIdx);
	void updateView();
	
	template <typename T>
	T* getChildView(const int row, const int column);

	/* member variable */
	bool splitted = false;
	bool maximized = false;

private:
	/* member function */
	void __maximizeActiveView();

	/* member variable */
	float __columnRatio = 0.5f;
	float __rowRatio = 0.5f;
	int __maximizedViewIdx;
	CRect __clientWindow;

public:
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
};

template <typename T>
T* CCustomSplitterWnd::getChildView(const int row, const int column)
{
	return static_cast<T*>(GetPane(row, column));
}