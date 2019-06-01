#include "AnchorManager.h"

AnchorManager::AnchorManager(
	const float &samplingStep_top,
	const float &samplingStep_front,
	const float &samplingStep_right) :
	__samplingStep_top(samplingStep_top),
	__samplingStep_front(samplingStep_front),
	__samplingStep_right(samplingStep_right)
{}

void AnchorManager::__sync()
{
	if (__dirty_top)
	{
		__anchor_top.x = (__volHalfSize.width + __anchorAdj_top.x);
		__anchor_top.y = (__volHalfSize.height + __anchorAdj_top.y);
		__dirty_top = false;
	}

	if (__dirty_front)
	{
		__anchor_front.x = (__volHalfSize.width + __anchorAdj_front.x);
		__anchor_front.y = (__volHalfSize.depth + __anchorAdj_front.y);
		__dirty_front = false;
	}

	if (__dirty_right)
	{
		__anchor_right.x = (__volHalfSize.height + __anchorAdj_right.x);
		__anchor_right.y = (__volHalfSize.depth + __anchorAdj_right.y);
		__dirty_right = false;
	}
}

void AnchorManager::init(const Size3D<> &volumeSize)
{
	__volHalfSize = (volumeSize.castTo<float>() * .5f);

	__anchorAdj_top.set(0.f, 0.f);
	__anchorAdj_front.set(0.f, 0.f);
	__anchorAdj_right.set(0.f, 0.f);

	__dirty_top = true;
	__dirty_front = true;
	__dirty_right = true;
}

void AnchorManager::adjustAnchor(const float deltaHoriz, const float deltaVert, const SliceAxis axis)
{
	switch (axis)
	{
	case SliceAxis::TOP:
		__dirty_top = true;
		__anchorAdj_top.x += (deltaHoriz * __samplingStep_top);
		__anchorAdj_top.y += (deltaVert * __samplingStep_top);
		break;

	case SliceAxis::FRONT:
		__dirty_front = true;
		__anchorAdj_front.x += (deltaHoriz * __samplingStep_front);
		__anchorAdj_front.y += (deltaVert * __samplingStep_front);
		break;

	case SliceAxis::RIGHT:
		__dirty_right = true;
		__anchorAdj_right.x += (deltaHoriz * __samplingStep_right);
		__anchorAdj_right.y += (deltaVert * __samplingStep_right);
		break;
	}
}

const Point2D &AnchorManager::getAnchorAdj(const SliceAxis axis) const
{
	switch (axis)
	{
	case SliceAxis::TOP:
		return __anchorAdj_top;

	case SliceAxis::FRONT:
		return __anchorAdj_front;

	case SliceAxis::RIGHT:
		return __anchorAdj_right;
	}

	return __anchorAdj_top;
}

const Point2D &AnchorManager::getAnchor(const SliceAxis axis)
{
	__sync();

	switch (axis)
	{
	case SliceAxis::TOP:
		return __anchor_top;

	case SliceAxis::FRONT:
		return __anchor_front;

	case SliceAxis::RIGHT:
		return __anchor_right;
	}

	return __anchor_top;
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
			SCR_SIZE_HALF.width - ((__anchorAdj_top.x + slicingPointAdj.x) / __samplingStep_top));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((__anchorAdj_top.y + slicingPointAdj.y) / __samplingStep_top));

		break;

	case SliceAxis::FRONT:
		retVal.x = static_cast<int>(
			SCR_SIZE_HALF.width - ((__anchorAdj_front.x + slicingPointAdj.x) / __samplingStep_front));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((__anchorAdj_front.y + slicingPointAdj.z) / __samplingStep_front));

		break;

	case SliceAxis::RIGHT:
		retVal.x = static_cast<int>(
			SCR_SIZE_HALF.width - ((__anchorAdj_right.x + slicingPointAdj.y) / __samplingStep_right));

		retVal.y = static_cast<int>(
			SCR_SIZE_HALF.height + ((__anchorAdj_right.y + slicingPointAdj.z) / __samplingStep_right));

		break;
	}

	return retVal;
}