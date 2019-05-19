/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	파일명			: NumberUtility.hpp
*	작성자			: 원진, 이세인
*	최종 수정일		: 19.03.22
*/

#include "NumberUtility.hpp"
#include <cmath>

namespace NumberUtility 
{
	/* function */
	__host__ __device__
	bool nearEqual(
		const float operand1, const float operand2, const float epsilon) 
	{
		return (fabsf(operand1 - operand2) < epsilon);
	}

	__host__ __device__
	Vector3D inverseGradient(
		const float operand1X, const float operand2X,
		const float operand1Y, const float operand2Y,
		const float operand1Z, const float operand2Z) 
	{
		const float X = (operand1X - operand2X);
		const float Y = (operand1Y - operand2Y);
		const float Z = (operand1Z - operand2Z);

		Vector3D retVal(X, Y, Z);

		if (!retVal.isZero())
			retVal.normalize();

		return retVal;
	}
}