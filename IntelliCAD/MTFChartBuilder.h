#pragma once

#include "stdafx.h"
#include <vector>
#include "Size2D.hpp"
#include "tstring.h"
#include "TFChartBuilder.h"

// Mono Transfer function Chart Builder
class MTFChartBuilder : public TFChartBuilder
{
private:
	std::vector<double> __data;
	int __numData = 0;

	XYChart *__pChart = nullptr;

	int __chainPrevDensity;
	double __chainPrevValue;
	bool __chainUpdate = false;

	int __datalineColor = RGB(70, 70, 70);

	void __release();
	void __build(const int canvasWidth, const int canvasHeight);

public:
	MTFChartBuilder(CChartViewer &chartViewer);

	std::tstring xAxisTitle = _T("Density");
	std::tstring yAxisTitle = _T("Value");
	std::tstring datalineName = _T("Filter");

	void setDatalineColor(const ubyte red, const ubyte green, const ubyte blue);

	void initData(const std::vector<double> &data);

	CPoint getMousePositionInPlot() const;
	void dragPlot(const CPoint &dragTo);
	void finishDragging();

	void renderChart(const int canvasWidth, const int canvasHeight);
	void renderPlotCrosshair(const CPoint &anchor);
};
