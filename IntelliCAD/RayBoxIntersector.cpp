/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: RayBoxIntersector.cpp
*	작성자			: 원진
*	최종 수정일		: 19.03.18
*/

#include "RayBoxInstersector.h"
#include "NumberUtility.hpp"
#include <limits>

namespace RayBoxIntersector
{
	/* function */
	__host__ __device__
	Range<float> getValidRange(
		const Size3D<float>& volumeSize,
		const Point3D& pixelPosition,
		const Vector3D& camDirection) 
	{
		// 보간에 의한 추가 영역 참조가 발생할 때,
		// 인덱스가 볼륨 영역 외부를 벗어나지 않도록 방지하기 위한 패딩 값
		const float PADDING = 1.f;

		const float VALID_WIDTH = ((volumeSize.width - 1.f) - PADDING);
		const float VALID_HEIGHT = ((volumeSize.height - 1.f) - PADDING);
		const float VALID_DEPTH = ((volumeSize.depth - 1.f) - PADDING);

		float start = 0.f;
		float end = FLT_MAX;

		const bool ZERO_DIR_X = NumberUtility::nearEqual(camDirection.x, 0.f);
		const bool ZERO_DIR_Y = NumberUtility::nearEqual(camDirection.y, 0.f);
		const bool ZERO_DIR_Z = NumberUtility::nearEqual(camDirection.z, 0.f);

		if (ZERO_DIR_X &&
			((pixelPosition.x < PADDING) || (pixelPosition.x > VALID_WIDTH)))
			return { 0.f, -1.f };

		if (ZERO_DIR_Y &&
			((pixelPosition.y < PADDING) || (pixelPosition.y > VALID_HEIGHT)))
			return { 0.f, -1.f };

		if (ZERO_DIR_Z &&
			((pixelPosition.z > PADDING) && (pixelPosition.z < VALID_DEPTH)))
			return { 0.f, -1.f };

		if (!ZERO_DIR_X)
		{
			const float INV_DIR_X = (1.f / camDirection.x);

			const float X1 = ((VALID_WIDTH - pixelPosition.x) * INV_DIR_X);
			const float X2 = ((PADDING - pixelPosition.x) * INV_DIR_X);

			float min = MIN(X1, X2);
			float max = MAX(X1, X2);

			start = MAX(start, min);
			end = MIN(end, max);
		}

		if (!ZERO_DIR_Y)
		{
			const float INV_DIR_Y = (1.f / camDirection.y);

			const float Y1 = ((VALID_HEIGHT - pixelPosition.y) * INV_DIR_Y);
			const float Y2 = ((PADDING - pixelPosition.y) * INV_DIR_Y);

			float min = MIN(Y1, Y2);
			float max = MAX(Y1, Y2);

			start = MAX(start, min);
			end = MIN(end, max);
		}

		if (!ZERO_DIR_Z)
		{
			const float INV_DIR_Z = (1.f / camDirection.z);

			const float Z1 = ((VALID_DEPTH - pixelPosition.z) * INV_DIR_Z);
			const float Z2 = ((PADDING - pixelPosition.z) * INV_DIR_Z);

			float min = MIN(Z1, Z2);
			float max = MAX(Z1, Z2);

			start = MAX(start, min);
			end = MIN(end, max);
		}

		return { start, end };
	}
}