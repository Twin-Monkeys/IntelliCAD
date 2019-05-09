#pragma once

#include "GPUVolume.h"
#include "Pixel.h"
#include "Size2D.hpp"

class ImageProcessor
{
public:
	void setVolume(const GPUVolume *const pVolume);
	void render(Pixel *const pScreen, const int screenWidth, const int screenHeight);
};