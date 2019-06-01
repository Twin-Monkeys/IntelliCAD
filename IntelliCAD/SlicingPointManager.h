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
	const float &__samplingStep_top;
	const float &__samplingStep_front;
	const float &__samplingStep_right;

	Size3D<float> __volHalfSize;

	Point3D __slicingPoint;
	Point3D __slicingPointAdj;

	bool __dirty;

	void __sync();

public:
	SlicingPointManager(
		const float &samplingStep_top,
		const float &samplingStep_front,
		const float &samplingStep_right);

	void init(const Size3D<> &volumeSize);

	void setSlicingPointFromScreen(
		const Size2D<> &screenSize, const Index2D<> &screenIdx, const Point2D &anchorAdj, const SliceAxis axis);

	void adjustSlicingPoint(const float delta, const SliceAxis axis);

	const Point3D &getSlicingPointAdj() const;
	const Point3D &getSlicingPoint();
};
