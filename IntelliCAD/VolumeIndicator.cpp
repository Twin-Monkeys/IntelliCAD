#include "VolumeIndicator.h"
#include "NumberUtility.hpp"
#include "MacroTransaction.h"

namespace VolumeIndicator
{
	__device__
	static bool __check_transaction(
		const float zeroTest1, const float zeroTest2, const float lengthTest,
		const float lineLength, const float zeroPrecision, const bool inverseLine)
	{
		IF_F_RET_F(NumberUtility::nearEqual(zeroTest1, 0.f, zeroPrecision));
		IF_F_RET_F(NumberUtility::nearEqual(zeroTest2, 0.f, zeroPrecision));

		if (!inverseLine)
			return NumberUtility::isInOfBound(lengthTest, -zeroPrecision, lineLength + zeroPrecision);

		return NumberUtility::isInOfBound(lengthTest, -(lineLength + zeroPrecision), zeroPrecision);
	}

	__device__
	static bool __recognize_axis(
		const float zeroTest1, const float zeroTest2, const float lengthTest,
		const float zeroTest1Stride, const float zeroTest2Stride, const float lengthTestStride,
		const float lineLength, const float zeroPrecision)
	{
		// y = 0, z = 0, 0 <= x <= lineLength
		IF_T_RET_T(__check_transaction(
			zeroTest1, zeroTest2, lengthTest,
			lineLength, zeroPrecision, false));

		// y = 0, z = 0, (VOL_SIZE.width - lineLength) <= x <= VOL_SIZE.width
		IF_T_RET_T(__check_transaction(
			zeroTest1, zeroTest2, lengthTest - lengthTestStride,
			lineLength, zeroPrecision, true));

		// y = VOL_SIZE.height, z = 0, 0 <= x <= lineLength
		IF_T_RET_T(__check_transaction(
			zeroTest1 - zeroTest1Stride, zeroTest2, lengthTest,
			lineLength, zeroPrecision, false));

		// y = VOL_SIZE.height, z = 0, (VOL_SIZE.width - lineLength) <= x <= VOL_SIZE.width
		IF_T_RET_T(__check_transaction(
			zeroTest1 - zeroTest1Stride, zeroTest2, lengthTest - lengthTestStride,
			lineLength, zeroPrecision, true));

		// y = 0, z = VOL_SIZE.depth, 0 <= x <= lineLength
		IF_T_RET_T(__check_transaction(
			zeroTest1, zeroTest2 - zeroTest2Stride, lengthTest,
			lineLength, zeroPrecision, false));

		// y = 0, z = VOL_SIZE.depth, (VOL_SIZE.width - lineLength) <= x <= VOL_SIZE.width
		IF_T_RET_T(__check_transaction(
			zeroTest1, zeroTest2 - zeroTest2Stride, lengthTest - lengthTestStride,
			lineLength, zeroPrecision, true));

		// y = VOL_SIZE.height, z = VOL_SIZE.depth, 0 <= x <= lineLength
		IF_T_RET_T(__check_transaction(
			zeroTest1 - zeroTest1Stride, zeroTest2 - zeroTest2Stride, lengthTest,
			lineLength, zeroPrecision, false));

		// y = VOL_SIZE.height, z = VOL_SIZE.depth, (VOL_SIZE.width - lineLength) <= x <= VOL_SIZE.width
		return __check_transaction(
			zeroTest1 - zeroTest1Stride, zeroTest2 - zeroTest2Stride, lengthTest - lengthTestStride,
			lineLength, zeroPrecision, true);
	}

	__device__
	bool recognize(const Size3D<float> &volSize, const Point3D &target, const float lineLength, const float lineThickness)
	{
		// X
		IF_T_RET_T(__recognize_axis(
			target.y, target.z, target.x,
			volSize.height, volSize.depth, volSize.width,
			lineLength, lineThickness));

		// Y
		IF_T_RET_T(__recognize_axis(
			target.z, target.x, target.y,
			volSize.depth, volSize.width, volSize.height,
			lineLength, lineThickness));

		// Z
		return __recognize_axis(
			target.x, target.y, target.z,
			volSize.width, volSize.height, volSize.depth,
			lineLength, lineThickness);
	}
}