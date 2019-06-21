#include "AnchorManager.h"
#include "System.h"

AnchorManager::AnchorManager(const float(&samplingStepArr)[3]) :
	__samplingStepArr(samplingStepArr)
{}

void AnchorManager::__sync(const SliceAxis axis)
{
	__anchorArr[axis].x = (__anchorBaseArr[axis].x + __anchorAdjArr[axis].x);
	__anchorArr[axis].y = (__anchorBaseArr[axis].y + __anchorAdjArr[axis].y);
}

void AnchorManager::init(const Size3D<float> &volumeSize)
{
	const Size3D<float> VOL_HALF_SIZE = (volumeSize * .5f);

	// Top
	__anchorBaseArr[SliceAxis::TOP].x = VOL_HALF_SIZE.width;
	__anchorBaseArr[SliceAxis::TOP].y = VOL_HALF_SIZE.height;

	// Front
	__anchorBaseArr[SliceAxis::FRONT].x = VOL_HALF_SIZE.width;
	__anchorBaseArr[SliceAxis::FRONT].y = VOL_HALF_SIZE.depth;

	// Right
	__anchorBaseArr[SliceAxis::RIGHT].x = VOL_HALF_SIZE.height;
	__anchorBaseArr[SliceAxis::RIGHT].y = VOL_HALF_SIZE.depth;

	EventBroadcaster &eventBroadcaster = System::getSystemContents().getEventBroadcaster();

	for (int i = 0; i < 3; i++)
	{
		const SliceAxis AXIS = static_cast<SliceAxis>(i);

		__anchorAdjArr[AXIS].set(0.f, 0.f);
		__sync(AXIS);

		eventBroadcaster.notifyInitSliceViewAnchor(AXIS);
	}
}

void AnchorManager::adjustAnchor(const float deltaX, const float deltaY, const SliceAxis axis)
{
	__anchorAdjArr[axis].x += (deltaX * __samplingStepArr[axis]);
	__anchorAdjArr[axis].y += (deltaY * __samplingStepArr[axis]);
	__sync(axis);
}

void AnchorManager::setAnchorAdj(const float adjX, const float adjY, const SliceAxis axis)
{
	__anchorAdjArr[axis].x = adjX;
	__anchorAdjArr[axis].y = adjY;
	__sync(axis);
}

const Point2D &AnchorManager::getAnchorAdj(const SliceAxis axis) const
{
	return __anchorAdjArr[axis];
}

const Point2D &AnchorManager::getAnchor(const SliceAxis axis) const
{
	return __anchorArr[axis];
}

Index2D<> AnchorManager::getSlicingPointForScreen(
	const Size2D<> &screenSize, const Point3D &slicingPointAdj, const SliceAxis axis)
{
	Index2D<> retVal;

	const Size2D<float> SCR_SIZE_HALF = (screenSize.castTo<float>() / 2.f);

	switch (axis)
	{
	case SliceAxis::TOP:
		retVal.x = static_cast<int>(
			SCR_SIZE_HALF.width + ((slicingPointAdj.x - __anchorAdjArr[axis].x) / __samplingStepArr[axis]));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((__anchorAdjArr[axis].y - slicingPointAdj.y) / __samplingStepArr[axis]));

		break;

	case SliceAxis::FRONT:
		retVal.x = static_cast<int>(
			SCR_SIZE_HALF.width + ((slicingPointAdj.x - __anchorAdjArr[axis].x) / __samplingStepArr[axis]));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((slicingPointAdj.z + __anchorAdjArr[axis].y) / __samplingStepArr[axis]));

		break;

	case SliceAxis::RIGHT:
		retVal.x = static_cast<int>(
			SCR_SIZE_HALF.width + ((slicingPointAdj.y - __anchorAdjArr[axis].x) / __samplingStepArr[axis]));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((__anchorAdjArr[axis].y + slicingPointAdj.z) / __samplingStepArr[axis]));

		break;
	}

	return retVal;
}