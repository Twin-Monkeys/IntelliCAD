#pragma once

#include "GPUVolume.h"
#include "Pixel.h"
#include "Size2D.hpp"
#include "SetVolumeListener.h"

class RenderingEngine : public SetVolumeListener
{
private:
	const GPUVolume *__pVolume = nullptr;

public:
	RenderingEngine();

	void setVolume(const GPUVolume *const pVolume);
	void render(Pixel *const pScreen, const int screenWidth, const int screenHeight);

	virtual void onSetVolume(const GPUVolume *const pVolume) override;
};
