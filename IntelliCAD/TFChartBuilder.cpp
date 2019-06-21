#include "TFChartBuilder.h"

TFChartBuilder::TFChartBuilder(CChartViewer &chartViewer) :
	_chartViewer(chartViewer)
{}

void TFChartBuilder::setBackgroundColor(
	const ubyte redStart, const ubyte greenStart, const ubyte blueStart,
	const ubyte redEnd, const ubyte greenEnd, const ubyte blueEnd)
{
	_bgStartColor = BGR(blueStart, greenStart, redStart);
	_bgEndColor = BGR(blueEnd, greenEnd, redEnd);
}

void TFChartBuilder::setPlotBackgroundColor(const ubyte red, const ubyte green, const ubyte blue)
{
	_plotBgColor = BGR(blue, green, red);
}

void TFChartBuilder::setHorizGridColor(const ubyte red, const ubyte green, const ubyte blue)
{
	_horizGridColor = BGR(blue, green, red);
}

void TFChartBuilder::setVertGridColor(const ubyte red, const ubyte green, const ubyte blue)
{
	_vertGridColor = BGR(blue, green, red);
}