#include "SlicingPointManager.h"
#include "NumberUtility.hpp"

SlicingPointManager::SlicingPointManager(
	const float &samplingStep_top,
	const float &samplingStep_front,
	const float &samplingStep_right) :
	__samplingStep_top(samplingStep_top),
	__samplingStep_front(samplingStep_front),
	__samplingStep_right(samplingStep_right)
{}

void SlicingPointManager::__sync()
{
	if (__dirty)
	{
		__slicingPoint.x = (__volHalfSize.width + __slicingPointAdj.x);
		__slicingPoint.y = (__volHalfSize.height + __slicingPointAdj.y);
		__slicingPoint.z = (__volHalfSize.depth + __slicingPointAdj.z);

		__dirty = false;
	}
}

void SlicingPointManager::init(const Size3D<> &volumeSize)
{
	__volHalfSize = (volumeSize.castTo<float>() * .5f);

	__slicingPointAdj.set(0.f, 0.f, 0.f);

	__dirty = true;
}

void SlicingPointManager::setSlicingPointFromScreen(
	const Size2D<> &screenSize, const Index2D<> &screenIdx, const Point2D &anchorAdj, const SliceAxis axis)
{
	const Size2D<float> SCR_SIZE_HALF = (screenSize.castTo<float>() / 2.f);
	const Index2D<float> SCR_IDX = screenIdx.castTo<float>();

	switch (axis)
	{
	case SliceAxis::TOP:
		__slicingPointAdj.x = ((SCR_SIZE_HALF.width - SCR_IDX.x) * __samplingStep_top - anchorAdj.x);
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		__slicingPointAdj.y = ((SCR_IDX.y - SCR_SIZE_HALF.height) * __samplingStep_top - anchorAdj.y);
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		break;

	case SliceAxis::FRONT:
		__slicingPointAdj.x = ((SCR_SIZE_HALF.width - SCR_IDX.x) * __samplingStep_front - anchorAdj.x);
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		__slicingPointAdj.z = ((SCR_IDX.y - SCR_SIZE_HALF.height) * __samplingStep_front - anchorAdj.y);
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);
		break;

	case SliceAxis::RIGHT:
		__slicingPointAdj.y = ((SCR_SIZE_HALF.width - SCR_IDX.x) * __samplingStep_top - anchorAdj.x);
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		__slicingPointAdj.z = ((SCR_IDX.y - SCR_SIZE_HALF.height) * __samplingStep_top - anchorAdj.y);
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);
		break;
	}

	__dirty = true;
}

void SlicingPointManager::adjustSlicingPoint(const float delta, const SliceAxis axis)
{
	switch (axis)
	{
	case SliceAxis::TOP:
		__slicingPointAdj.z += delta;
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);

		break;

	case SliceAxis::FRONT:
		__slicingPointAdj.y += delta;
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		break;

	case SliceAxis::RIGHT:
		__slicingPointAdj.x -= delta;
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		break;
	}

	__dirty = true;
}

const Point3D &SlicingPointManager::getSlicingPointAdj() const
{
	return __slicingPointAdj;
}

const Point3D &SlicingPointManager::getSlicingPoint()
{
	__sync();

	return __slicingPoint;
}