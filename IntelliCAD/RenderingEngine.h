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
#include "SystemInitListener.h"
#include "VolumeLoadingListener.h"
#include "TransferFunction.h"
#include "Camera.h"
#include "Light.h"
#include "SliceAxis.h"

class RenderingEngine : public SystemInitListener, public VolumeLoadingListener
{
public:
	class VolumeRenderer
	{
	public:
		/* member function */
		void render(Pixel* const pScreen, const int screenWidth, const int screenHeight);
		void adjustImgBasedSamplingStep(const float delta);

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
	private:
		friend RenderingEngine;

		/* member variable */
		const VolumeMeta& __volumeMeta;
		const bool &__initialized;

		float *__pHostTransferTableBuffer = nullptr;
		cudaArray_t __transferTableBuffer = nullptr;

		Range<ushort> _transferTableBoundary;
		bool __transferTableDirty;

		Point3D __slicingPoint;

		/* constructor */
		ImageProcessor(const VolumeMeta& volumeMeta, const bool &initFlag);

		/* member function */
		void __onLoadVolume();

		void __sync();
		void __syncTransferFunction();

		void __init();
		void __release();

	public:
		/* member function */
		void setTransferFunction(const ushort startInc, const ushort endExc);
		void setTransferFunction(const Range<ushort> &boundary);

		void render(
			Pixel *const pScreen, const int screenWidth, const int screenHeight,
			const float samplingStep, const SliceAxis axis);

		~ImageProcessor();
	};

	/* member function */
	void loadVolume(const VolumeData &volumeData);

	virtual void onSystemInit() override;
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
