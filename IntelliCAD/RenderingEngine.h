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
#include "SlicingPointManager.h"
#include "AnchorManager.h"

class RenderingEngine : public SystemInitListener, public VolumeLoadingListener
{
public:
	class VolumeRenderer
	{
	public:
		/* member function */
		void render(Pixel* const pScreen, const int screenWidth, const int screenHeight);
		void adjustImgBasedSamplingStep(const float delta);

		void setIndicatorLength(const float length);
		void setIndicatorThickness(const float thickness);

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

		~VolumeRenderer();

		/* member variable */
		TransferFunction* __pTransferFunc = nullptr;
		Light __lights[3];
		float __imgBasedSamplingStep;
		float __objectBasedSamplingStep;
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

		Size3D<float> volumeHalfSize;

		SlicingPointManager __slicingPointMgr;
		AnchorManager __anchorMgr;

		float __samplingStep_top;
		float __samplingStep_front;
		float __samplingStep_right;

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
		Index2D<> getSlicingPointForScreen(const Size2D<> &screenSize, const SliceAxis axis);

		void setTransferFunction(const ushort startInc, const ushort endExc);
		void setTransferFunction(const Range<ushort> &boundary);

		void setSlicingPointFromScreen(const Size2D<> &screenSize, const Index2D<> &screenIdx, const SliceAxis axis);
		void adjustSlicingPoint(const float delta, const SliceAxis axis);
		
		void adjustSamplingStep(const float delta, const SliceAxis axis);
		void adjustAnchor(const float deltaHoriz, const float deltaVert, const SliceAxis axis);

		void render(
			Pixel *const pScreen, const int screenWidth, const int screenHeight, const SliceAxis axis);

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
