#include <sstream>
#include "CTFChartBuilder.h"
#include "System.h"
#include "Parser.hpp"
#include "NumberUtility.hpp"

#define BGR(b, g, r) RGB(b, g, r)

using namespace std;

CTFChartBuilder::CTFChartBuilder(CChartViewer &chartViewer) :
	TFChartBuilder(chartViewer)
{}

void CTFChartBuilder::__release()
{
	if (__pChart)
	{
		delete __pChart;
		__pChart = nullptr;
	}
}

int CTFChartBuilder::__getFirstActiveColorIndex()
{
	int retVal = -1;

	for (int i = 0; i < 4; i++)
		if (activeColorFlagArr[i])
		{
			retVal = i;
			break;
		}

	return retVal;
}

void CTFChartBuilder::setDatalineColor(
	const ColorChannelType colorType, const ubyte red, const ubyte green, const ubyte blue)
{
	__datalineColorArr[static_cast<int>(colorType)] = BGR(blue, green, red);
}

void CTFChartBuilder::toggleColorActivation(const ColorChannelType colorType)
{
	if (colorType == ColorChannelType::ALL)
	{
		for (int i = 0; i < 4; i++)
			activeColorFlagArr[i] = !activeColorFlagArr[i];

		return;
	}

	activeColorFlagArr[colorType] = !activeColorFlagArr[colorType];
}

void CTFChartBuilder::setColorActivation(const ColorChannelType colorType, const bool flag)
{
	if (colorType == ColorChannelType::ALL)
	{
		for (int i = 0; i < 4; i++)
			activeColorFlagArr[i] = flag;

		return;
	}

	activeColorFlagArr[colorType] = flag;
}

void CTFChartBuilder::initData(const ColorChannelType colorType, const vector<double> &data)
{
	__dataArr[colorType] = data;
	__numData = static_cast<int>(data.size());
}

void CTFChartBuilder::initData(
	const vector<double> &dataRed, const vector<double> &dataGreen,
	const vector<double> &dataBlue, const vector<double> &dataAlpha)
{
	__dataArr[0] = dataRed;
	__dataArr[1] = dataGreen;
	__dataArr[2] = dataBlue;
	__dataArr[3] = dataAlpha;

	__numData = static_cast<int>(dataRed.size());
}

CPoint CTFChartBuilder::getMousePositionInPlot() const
{
	CPoint retVal;
	retVal.x = _chartViewer.getPlotAreaMouseX();
	retVal.y = _chartViewer.getPlotAreaMouseY();

	return retVal;
}

void CTFChartBuilder::dragPlot(const CPoint &dragTo)
{
	const int FIRST_ACTIVE_COLOR_IDX = __getFirstActiveColorIndex();
	if (FIRST_ACTIVE_COLOR_IDX == -1)
		return;

	const int DENSITY = static_cast<int>(__pChart->getXValue(dragTo.x));
	const double VALUE = __pChart->getYValue(dragTo.y, __pChart->yAxis());

	double *const pFirstActive = __dataArr[FIRST_ACTIVE_COLOR_IDX].data();

	int modifiedFrom = 0;
	int modifiedMemCount = 0;
	
	if (!__chainUpdate)
	{
		__chainUpdate = true;

		if ((DENSITY >= 0) && (DENSITY < __numData))
		{
			pFirstActive[DENSITY] = VALUE;
			modifiedMemCount = 1;
		}

		modifiedFrom = DENSITY;
	}
	else
	{
		const double ITER = (DENSITY - __chainPrevDensity);
		const double LEVEL = ((VALUE - __chainPrevValue) / ITER);

		if (__chainPrevDensity < DENSITY)
		{
			const int FROM = max(__chainPrevDensity + 1, 0);
			const int TO = min(DENSITY, __numData - 1);

			for (int i = FROM; i <= TO; i++, modifiedMemCount++)
				pFirstActive[i] = (__chainPrevValue + ((i - __chainPrevDensity) * LEVEL));

			modifiedFrom = FROM;
		}
		else
		{
			const int FROM = max(DENSITY, 0);
			const int TO = min(__chainPrevDensity - 1, __numData - 1);

			for (int i = TO; i >= FROM; i--, modifiedMemCount++)
				pFirstActive[i] = (__chainPrevValue - ((__chainPrevDensity - i) * LEVEL));

			modifiedFrom = FROM;
		}
	}

	if (modifiedMemCount)
	{
		for (int i = (FIRST_ACTIVE_COLOR_IDX + 1); i < 4; i++)
		{
			if (activeColorFlagArr[i])
				memcpy(__dataArr[i].data() + modifiedFrom,
					pFirstActive + modifiedFrom, modifiedMemCount * sizeof(double));
		}
	}
	
	__chainPrevDensity = DENSITY;
	__chainPrevValue = VALUE;
}

void CTFChartBuilder::finishDragging()
{
	__chainUpdate = false;

	for (int i = 0; i < 4; i++)
	{
		if (activeColorFlagArr[i])
		{
			System::getSystemContents().getRenderingEngine().volumeRenderer.
				setTransferFunction(static_cast<ColorChannelType>(i), __dataArr[i].data());
		}
	}
}

void CTFChartBuilder::__build(const int canvasWidth, const int canvasHeight)
{
	__release();

	// new XYChart(차트가 그려질 전체 영역), 10은 하드코딩 스타일의 마진 조정
	__pChart = new XYChart(canvasWidth, canvasHeight + 9);

	// 차트의 배경색 설정 (그라디언트)
	__pChart->setBackground(
		__pChart->linearGradientColor(
			0, 0, 0, __pChart->getHeight() / 2, _bgStartColor, _bgEndColor));

	const Size2D<> PLOT_SIZE =
	{
		canvasWidth - ((2 * padding.width)),
		canvasHeight - ((2 * padding.height) + 20)	// 마진 조정
	};

	// 차트 플롯이 그려질 영역 (좌상단 꼭지점, 크기)
	PlotArea *const pPlotArea = __pChart->setPlotArea(
		padding.width + 13, padding.height, PLOT_SIZE.width, PLOT_SIZE.height, _plotBgColor, -1, -1,
		__pChart->dashLineColor(_horizGridColor, Chart::DotLine),
		__pChart->dashLineColor(_vertGridColor, Chart::DotLine));

	// 범례 추가
	__pChart->addLegend(70, 13, false, "arialbd.ttf", 8)->setBackground(Chart::Transparent);

	// 선형 그래프를 그린다. (그래프 데이터를 함께 입력함)
	LineLayer *const pLayer = __pChart->addLineLayer();
	pLayer->setLineWidth(2);

	for (int i = 0; i < 4; i++)
	{
		pLayer->addDataSet(
			DoubleArray(__dataArr[i].data(), __numData),
			__datalineColorArr[i], Parser::tstring$string(datalineNameArr[i]).c_str());
	}

	// X축에 관한 정보를 다룰 수 있는 XAxis 객체
	XAxis *const pXAxis = __pChart->xAxis();
	pXAxis->setLinearScale(0., static_cast<double>(__numData - 1));

	// Tick 캡션 설정
	pXAxis->setTitle(Parser::tstring$string(xAxisTitle).c_str());

	YAxis *const pYAxis = __pChart->yAxis();
	pYAxis->setLinearScale(0., 1.);
	pYAxis->setTitle(Parser::tstring$string(yAxisTitle).c_str());
}

void CTFChartBuilder::renderChart(const int canvasWidth, const int canvasHeight)
{
	__build(canvasWidth, canvasHeight);
	_chartViewer.setChart(__pChart);
}

void CTFChartBuilder::renderPlotCrosshair(const CPoint &anchor)
{
	PlotArea *const pPlotArea = __pChart->getPlotArea();

	// Crosshair를 그리기 위한 추가 레이어 생성
	DrawArea *const pDrawArea = __pChart->initDynamicLayer();

	// 가로 축 그리기
	pDrawArea->hline(
		pPlotArea->getLeftX(), pPlotArea->getRightX(),
		anchor.y, pDrawArea->dashLineColor(BGR(10, 10, 10)));

	// 세로 축 그리기
	pDrawArea->vline(
		pPlotArea->getTopY(), pPlotArea->getBottomY(),
		anchor.x, pDrawArea->dashLineColor(BGR(10, 10, 10)));

	// 레이블 달기
	ostringstream labelStream;
	const double INDEX = __pChart->getXValue(anchor.x);
	const double VALUE = __pChart->getYValue(anchor.y, __pChart->yAxis());

	// X축 레이블
	labelStream <<
		"<*block,bgColor=FFFFDD,margin=4,edgeColor=0A0A0A*>" <<
		__pChart->formatValue(INDEX, "{value|P0}") <<
		"<*/*>";

	TTFText *const pXLabelBox = pDrawArea->text(labelStream.str().c_str(), "arialbd.ttf", 8);
	pXLabelBox->draw(anchor.x, pPlotArea->getBottomY() + 5, BGR(10, 10, 10), Chart::Top);
	pXLabelBox->destroy();

	// Y축 레이블
	labelStream.str("");
	labelStream <<
		"<*block,bgColor=FFFFDD,margin=4,edgeColor=0A0A0A*>" <<
		__pChart->formatValue(VALUE, "{value|P2}") <<
		"<*/*>";

	TTFText *const pYLabelBox = pDrawArea->text(labelStream.str().c_str(), "arialbd.ttf", 8);
	pYLabelBox->draw(pPlotArea->getLeftX() - 5, anchor.y, BGR(10, 10, 10), Chart::Right);
	pYLabelBox->destroy();

	_chartViewer.updateDisplay();
	_chartViewer.removeDynamicLayer(CVN_MouseLeavePlotArea);
}