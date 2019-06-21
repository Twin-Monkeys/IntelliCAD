#pragma once

#include "Size3D.hpp"
#include "Point2D.h"
#include "SliceAxis.h"
#include "Index2D.hpp"
#include "Point3D.h"
#include "Size2D.hpp"

class AnchorManager
{
private:
	const float(&__samplingStepArr)[3];

	// Top, Front, Right
	Point2D __anchorBaseArr[3];
	Point2D __anchorAdjArr[3];
	Point2D __anchorArr[3];

	void __sync(const SliceAxis axis);

public:
	AnchorManager(const float(&samplingStepArr)[3]);

	void init(const Size3D<float> &volumeSize);
	void adjustAnchor(const float deltaX, const float deltaY, const SliceAxis axis);
	void setAnchorAdj(const float adjX, const float adjY, const SliceAxis axis);

	const Point2D &getAnchorAdj(const SliceAxis axis) const;
	const Point2D &getAnchor(const SliceAxis axis) const;

	Index2D<> getSlicingPointForScreen(
		const Size2D<> &screenSize, const Point3D &slicingPointAdj, const SliceAxis axis);
};
