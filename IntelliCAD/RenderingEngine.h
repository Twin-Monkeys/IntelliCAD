#pragma once

#include "Pixel.h"
#include "Size2D.hpp"
#include "VolumeLoadingListener.h"

class RenderingEngine : public VolumeLoadingListener
{
private:
	static RenderingEngine __instance;
	VolumeMeta __volumeMeta;

	RenderingEngine();

public:
	class VolumeRenderer
	{
	private:
		friend RenderingEngine;
		const VolumeMeta &__volumeMeta;

		VolumeRenderer(const VolumeMeta &volumeMeta);

		void __onLoadVolume();

	public:
		void render(Pixel *const pScreen, const int screenWidth, const int screenHeight);
	};

	class ImageProcessor
	{
	private:
		friend RenderingEngine;
		const VolumeMeta &__volumeMeta;

		ImageProcessor(const VolumeMeta &volumeMeta);

		void __onLoadVolume();

	public:
		void render(Pixel *const pScreen, const int screenWidth, const int screenHeight);
	};

	VolumeRenderer volumeRenderer;
	ImageProcessor imageProcessor;

	void loadVolume(const VolumeData &volumeData);
	virtual void onLoadVolume(const VolumeData &volumeData) override;

	static RenderingEngine &getInstance();
};
