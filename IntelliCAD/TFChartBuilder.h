#pragma once

#define BGR(b, g, r) RGB(b, g, r)

#include "stdafx.h"
#include "ChartViewer.h"
#include "TypeEx.h"
#include "Size2D.hpp"

class TFChartBuilder
{
protected:
	CChartViewer &_chartViewer;

	int _bgStartColor = BGR(231, 220, 214);
	int _bgEndColor = BGR(251, 250, 240);

	int _plotBgColor = BGR(255, 255, 255);
	int _horizGridColor = BGR(190, 190, 190);
	int _vertGridColor = BGR(190, 190, 190);

public:
	TFChartBuilder(CChartViewer &chartViewer);

	Size2D<> padding = { 60, 40 };

	void setBackgroundColor(
		const ubyte redStart, const ubyte greenStart, const ubyte blueStart,
		const ubyte redEnd, const ubyte greenEnd, const ubyte blueEnd);

	void setPlotBackgroundColor(const ubyte red, const ubyte green, const ubyte blue);
	void setHorizGridColor(const ubyte red, const ubyte green, const ubyte blue);
	void setVertGridColor(const ubyte red, const ubyte green, const ubyte blue);
};
