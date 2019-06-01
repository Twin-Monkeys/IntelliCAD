/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: Constant.h
*	�ۼ���			: ����
*	���� ������		: 19.04.07
*/

#pragma once

#include "Point3D.h"
#include "Range.hpp"
#include "Color.hpp"
#include "tstring.h"

namespace Constant
{
	namespace Window
	{
		extern const char* const TITLE;
	}

	namespace Volume
	{
		extern const int WIDTH;
		extern const int HEIGHT;
		extern const int DEPTH;
		extern const Point3D PIVOT;
	}

	namespace Material
	{
		extern const float SHININESS;
		namespace TransferFunc
		{
			namespace Filter
			{
				extern const Range<int> SKIN;
				extern const Range<int> BONE;
			}
		}
	}

	namespace Light
	{
		namespace Position
		{
			extern const Point3D LEFT;
			extern const Point3D RIGHT;
			extern const Point3D FRONT;
			extern const Point3D BACK;
			extern const Point3D TOP;
			extern const Point3D BOTTOM;
		}
	}

	namespace Camera
	{
		extern const Vector3D EYE;
		extern const Vector3D AT;
		extern const Vector3D UP;
	}

	namespace Database
	{
		extern const std::tstring DB_ROOT;
	}
}