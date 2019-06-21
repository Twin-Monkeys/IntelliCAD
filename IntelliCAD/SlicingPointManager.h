#pragma once

#include "Size3D.hpp"
#include "Point3D.h"
#include "SliceAxis.h"
#include "Size2D.hpp"
#include "Index2D.hpp"
#include "Point2D.h"

class SlicingPointManager
{
private:
	const float(&__samplingStepArr)[3];

	Size3D<float> __volHalfSize;

	Point3D __slicingPoint;
	Point3D __slicingPointAdj;

	void __sync();

public:
	SlicingPointManager(const float(&samplingStepArr)[3]);

	void init(const Size3D<float> &volumeSize);

	void setSlicingPointFromScreen(
		const Size2D<> &screenSize, const Index2D<> &screenIdx, const Point2D &anchorAdj, const SliceAxis axis);

	void setSlicingPointAdj(const Point3D &adj);
	void setSlicingPoint(const Point3D &point);

	void adjustSlicingPoint(const float delta, const SliceAxis axis);

	const Point3D &getSlicingPointAdj() const;
	const Point3D &getSlicingPoint() const;
};
