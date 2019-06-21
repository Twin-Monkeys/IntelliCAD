#include <memory>
#include "Pixel.h"
#include "NumberUtility.hpp"

__host__ __device__
Pixel::Pixel(const ubyte r, const ubyte g, const ubyte b)
	: rgb{ r, g, b }
{}

__host__ __device__
Pixel::Pixel(const ubyte value)
	: rgb {value, value, value}
{}

__host__ __device__
void Pixel::set(const ubyte r, const ubyte g, const ubyte b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}

__host__ __device__
void Pixel::set(const float r, const float g, const float b)
{
	this->r = static_cast<ubyte>(NumberUtility::truncate(r * 255.f, 0.f, 255.f));
	this->g = static_cast<ubyte>(NumberUtility::truncate(g * 255.f, 0.f, 255.f));
	this->b = static_cast<ubyte>(NumberUtility::truncate(b * 255.f, 0.f, 255.f));
}

__host__ __device__
void Pixel::set(const ubyte value)
{
	memset(rgb, value, 3);
}

__host__ __device__
void Pixel::set(const float value)
{
	const ubyte VALUE =
		static_cast<ubyte>(NumberUtility::truncate(value * 255.f, 0.f, 255.f));

	memset(rgb, VALUE, 3);
}

__host__ __device__
void Pixel::set(const Pixel &another)
{
	memcpy(rgb, another.rgb, 3);
}