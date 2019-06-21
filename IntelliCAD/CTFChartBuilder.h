#pragma once

#include "stdafx.h"
#include <vector>
#include <memory>
#include "ChartViewer.h"
#include "Size2D.hpp"
#include "tstring.h"
#include "TFChartBuilder.h"
#include "ColorChannelType.h"

// Color Transfer function Chart Builder
class CTFChartBuilder : public TFChartBuilder
{
private:
	// R, G, B, A
	std::vector<double> __dataArr[4];
	int __numData = 0;

	XYChart *__pChart = nullptr;

	int __chainPrevDensity;
	double __chainPrevValue;
	bool __chainUpdate = false;

	// R, G, B, A
	int __datalineColorArr[4] =
	{
		RGB(30, 30, 200), RGB(30, 170, 30),
		RGB(180, 90, 90), RGB(70, 70, 70)
	};

	void __release();
	void __build(const int canvasWidth, const int canvasHeight);

	int __getFirstActiveColorIndex();

public:
	CTFChartBuilder(CChartViewer &chartViewer);

	bool activeColorFlagArr[4] = { true, true, true, true };

	std::tstring xAxisTitle = _T("Density");
	std::tstring yAxisTitle = _T("Value");

	// R, G, B, A
	std::tstring datalineNameArr[4] =
	{
		 _T("Red Filter"),  _T("Green Filter"),
		 _T("Blue Filter"),  _T("Alpha Filter")
	};

	void setDatalineColor(const ColorChannelType colorType, const ubyte red, const ubyte green, const ubyte blue);

	void toggleColorActivation(const ColorChannelType colorType);
	void setColorActivation(const ColorChannelType colorType, const bool flag);

	void initData(const ColorChannelType colorType, const std::vector<double> &data);
	void initData(
		const std::vector<double> &dataRed, const std::vector<double> &dataGreen,
		const std::vector<double> &dataBlue, const std::vector<double> &dataAlpha);

	CPoint getMousePositionInPlot() const;
	void dragPlot(const CPoint &dragTo);
	void finishDragging();

	void renderChart(const int canvasWidth, const int canvasHeight);
	void renderPlotCrosshair(const CPoint &anchor);
};
