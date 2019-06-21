#pragma once

#include <cuda_runtime.h>
#include "TypeEx.h"

__align__(4)
class Pixel
{
public:

	//// DO NOT MOVE VARIABLE POSITION ////
	union
	{
		struct
		{
			ubyte r;
			ubyte g;
			ubyte b;
		};

		ubyte rgb[3];
	};

	ubyte dummy = 255;

	Pixel() = default;

	__host__ __device__
	Pixel(const ubyte r, const ubyte g, const ubyte b);

	__host__ __device__
	Pixel(const ubyte value);

	__host__ __device__
	void set(const ubyte r, const ubyte g, const ubyte b);

	__host__ __device__
	void set(const float r, const float g, const float b);

	__host__ __device__
	void set(const ubyte value);

	__host__ __device__
	void set(const float value);

	__host__ __device__
	void set(const Pixel &another);
};