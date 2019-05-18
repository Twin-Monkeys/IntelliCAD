/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: RenderingEngine.h
*	작성자			: 원진
*	최종 수정일		: 19.04.07
*/

#pragma once

#include "Pixel.h"
#include "Size2D.hpp"
#include "VolumeLoadingListener.h"
#include "TransferFunction.h"
#include "Camera.h"
#include "Light.h"

class RenderingEngine : public VolumeLoadingListener
{
public:
	class VolumeRenderer
	{
	public:
		/* member function */
		void render(Pixel* const pScreen, const int screenWidth, const int screenHeight);

		/* member variable */
		Camera camera;

	private:
		friend RenderingEngine;

		/* constructor */
		VolumeRenderer(const VolumeMeta& volumeMeta, const bool &initFlag);

		/* member function */
		void __onLoadVolume();
		void __initTransferFunc();
		void __initLight();
		void __initCamera();

		/* member variable */
		TransferFunction* __pTransferFunc = nullptr;
		Light __light[3];
		float __imgBasedSamplingStep = 1.f;
		float __objectBasedSamplingStep = 1.f;
		float __shininess = 40.f;
		bool __transferFuncDirty = true;

		const VolumeMeta& __volumeMeta;
		const bool &__initialized;
	};

	class ImageProcessor
	{
	public:
		/* member function */
		void render(Pixel* const pScreen, const int screenWidth, const int screenHeight);

	private:
		/* constructor */
		ImageProcessor(const VolumeMeta& volumeMeta, const bool &initFlag);

		/* member function */
		void __onLoadVolume();

		/* member variable */
		friend RenderingEngine;
		const VolumeMeta& __volumeMeta;
		const bool &__initialized;
	};

	/* member function */
	void loadVolume(const VolumeData &volumeData);
	virtual void onLoadVolume(const VolumeData &volumeData) override;
	static RenderingEngine& getInstance();

	/* member variable */
	VolumeRenderer volumeRenderer;
	ImageProcessor imageProcessor;

private:
	/* constructor */
	RenderingEngine();

	/* member variable */
	static RenderingEngine __instance;
	VolumeMeta __volumeMeta;

	bool __initialized = false;
};
