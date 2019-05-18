/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: RayBoxIntersector.h
*	작성자			: 원진
*	최종 수정일		: 19.03.18
*/

#pragma once

#include "Point3D.h"
#include "Range.hpp"
#include "Size3D.hpp"

namespace RayBoxIntersector 
{
	/* function */
	/// <summary>
	/// <para>주어진 픽셀에서 카메라 시점 방향으로 시야 광선을 투사하였을 때</para>
	/// <para>볼륨을 투과하는지 여부를 조사하고, 투과 영역을 계산한다.</para>
	/// </summary>
	/// <param name="volumeSize">
	/// 볼륨 크기
	/// </param>
	/// <param name="pixelPosition">
	/// 스크린 픽셀 위치
	/// </param>
	/// <param name="camDirection">
	/// 카메라 시점 벡터
	/// </param>
	/// <returns>
	/// 투과 영역
	/// </returns>
	__host__ __device__
	Range<float> getValidRange(
		const Size3D<>& volumeSize,
		const Point3D& pixelPosition, 
		const Vector3D& camDirection);
}