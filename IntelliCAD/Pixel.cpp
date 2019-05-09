#include <memory>
#include "Pixel.h"

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
void Pixel::set(const ubyte value)
{
	memset(rgb, value, 3);
}

__host__ __device__
void Pixel::set(const Pixel &another)
{
	memcpy(rgb, another.rgb, 3);
}