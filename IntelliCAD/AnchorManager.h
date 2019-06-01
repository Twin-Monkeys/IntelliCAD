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
	const float &__samplingStep_top;
	const float &__samplingStep_front;
	const float &__samplingStep_right;

	Size3D<float> __volHalfSize;

	Point2D __anchorAdj_top;
	Point2D __anchorAdj_front;
	Point2D __anchorAdj_right;

	Point2D __anchor_top;
	Point2D __anchor_front;
	Point2D __anchor_right;

	bool __dirty_top;
	bool __dirty_front;
	bool __dirty_right;

	void __sync();

public:
	AnchorManager(
		const float &samplingStep_top,
		const float &samplingStep_front,
		const float &samplingStep_right);

	void init(const Size3D<> &volumeSize);
	void adjustAnchor(const float deltaHoriz, const float deltaVert, const SliceAxis axis);

	const Point2D &getAnchorAdj(const SliceAxis axis) const;
	const Point2D &getAnchor(const SliceAxis axis);

	Index2D<> getSlicingPointForScreen(
		const Size2D<> &screenSize, const Point3D &slicingPointAdj, const SliceAxis axis);
};
