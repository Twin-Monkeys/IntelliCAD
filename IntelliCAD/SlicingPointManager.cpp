#include "SlicingPointManager.h"
#include "NumberUtility.hpp"
#include "System.h"

SlicingPointManager::SlicingPointManager(const float(&samplingStepArr)[3]) :
	__samplingStepArr(samplingStepArr)
{}

void SlicingPointManager::__sync()
{
	__slicingPoint.x = (__volHalfSize.width + __slicingPointAdj.x);
	__slicingPoint.y = (__volHalfSize.height + __slicingPointAdj.y);
	__slicingPoint.z = (__volHalfSize.depth + __slicingPointAdj.z);
}

void SlicingPointManager::init(const Size3D<float> &volumeSize)
{
	__volHalfSize = (volumeSize * .5f);
	__slicingPointAdj.set(0.f, 0.f, 0.f);

	__sync();

	System::getSystemContents().
		getEventBroadcaster().notifyInitSlicingPoint(__slicingPoint);
}

void SlicingPointManager::setSlicingPointFromScreen(
	const Size2D<> &screenSize, const Index2D<> &screenIdx, const Point2D &anchorAdj, const SliceAxis axis)
{
	const Size2D<float> SCR_SIZE_HALF = (screenSize.castTo<float>() / 2.f);
	const Index2D<float> SCR_IDX = screenIdx.castTo<float>();

	switch (axis)
	{
	case SliceAxis::TOP:
		__slicingPointAdj.x = (((SCR_IDX.x - SCR_SIZE_HALF.width) * __samplingStepArr[axis]) + anchorAdj.x);
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		__slicingPointAdj.y = (((SCR_SIZE_HALF.height - SCR_IDX.y) * __samplingStepArr[axis]) + anchorAdj.y);
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		break;

	case SliceAxis::FRONT:
		__slicingPointAdj.x = (((SCR_IDX.x - SCR_SIZE_HALF.width) * __samplingStepArr[axis]) + anchorAdj.x);
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		__slicingPointAdj.z = (((SCR_IDX.y - SCR_SIZE_HALF.height) * __samplingStepArr[axis]) - anchorAdj.y);
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);
		break;

	case SliceAxis::RIGHT:
		__slicingPointAdj.y = (((SCR_IDX.x - SCR_SIZE_HALF.width) * __samplingStepArr[axis]) + anchorAdj.x);
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		__slicingPointAdj.z = (((SCR_IDX.y - SCR_SIZE_HALF.height) * __samplingStepArr[axis]) - anchorAdj.y);
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);
		break;
	}

	__sync();
}

void SlicingPointManager::setSlicingPointAdj(const Point3D &adj)
{
	__slicingPointAdj.set(
	NumberUtility::truncate(adj.x, -__volHalfSize.width, __volHalfSize.width),
	NumberUtility::truncate(adj.y, -__volHalfSize.height, __volHalfSize.height),
	NumberUtility::truncate(adj.z, -__volHalfSize.depth, __volHalfSize.depth));

	__sync();
}

void SlicingPointManager::setSlicingPoint(const Point3D &point)
{
	setSlicingPointAdj({
		point.x - __volHalfSize.width,
		point.y - __volHalfSize.height,
		point.z - __volHalfSize.depth
		});
}

void SlicingPointManager::adjustSlicingPoint(const float delta, const SliceAxis axis)
{
	switch (axis)
	{
	case SliceAxis::TOP:
		__slicingPointAdj.z -= delta;
		__slicingPointAdj.z =
			NumberUtility::truncate(__slicingPointAdj.z, -__volHalfSize.depth, __volHalfSize.depth);

		break;

	case SliceAxis::FRONT:
		__slicingPointAdj.y -= delta;
		__slicingPointAdj.y =
			NumberUtility::truncate(__slicingPointAdj.y, -__volHalfSize.height, __volHalfSize.height);

		break;

	case SliceAxis::RIGHT:
		__slicingPointAdj.x += delta;
		__slicingPointAdj.x =
			NumberUtility::truncate(__slicingPointAdj.x, -__volHalfSize.width, __volHalfSize.width);

		break;
	}

	__sync();
}

const Point3D &SlicingPointManager::getSlicingPointAdj() const
{
	return __slicingPointAdj;
}

const Point3D &SlicingPointManager::getSlicingPoint() const
{
	return __slicingPoint;
}