/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	파일명			: RenderingEngine.h
*	작성자			: 원진
*	최종 수정일		: 19.04.07
*/

#pragma once

#include <vector>
#include "Pixel.h"
#include "VolumeData.h"
#include "Size2D.hpp"
#include "Camera.h"
#include "Light.h"
#include "SliceAxis.h"
#include "SlicingPointManager.h"
#include "AnchorManager.h"
#include "MonoTransferFunction.h"
#include "ColorTransferFunction.h"
#include "SystemIndirectAccessor.h"
#include "MacroTransaction.h"

class RenderingEngine
{
public:
	class VolumeRenderer
	{
	public:
		/* member function */
		void render(Pixel* const pScreen, const int screenWidth, const int screenHeight);
		void adjustImgBasedSamplingStep(const float delta);

		bool initTransferFunction(const ColorChannelType colorType);
		void initLight();

		template <typename T>
		std::vector<T> getTransferFunctionAs(const ColorChannelType colorType) const;

		const Light &getLight(const int index) const;

		template <typename T>
		bool setTransferFunction(const ColorChannelType colorType, const T* const pData);

		void setIndicatorLength(const float length);
		void setIndicatorThickness(const float thickness);
		void toggleLighting(const int index);

		void setLightAmbient(const int index, const Color<float>& ambient);
		void setLightDiffuse(const int index, const Color<float>& diffuse);
		void setLightSpecular(const int index, const Color<float>& specular);

		void setLightXPos(const int index, const float x);
		void setLightYPos(const int index, const float y);
		void setLightZPos(const int index, const float z);

		/* member variable */
		Camera camera;

	private:
		friend RenderingEngine;

		/* constructor */
		VolumeRenderer(const VolumeNumericMeta& volumeNumericMeta, const bool &volumeLoadedFlag);

		/* member function */
		void __init();
		
		void __initCamera();
		void __initImgBasedSamplingStep();
		void __release();

		void __syncLight();
		void __syncTransferFunction(const ColorChannelType colorType);

		~VolumeRenderer();

		/* member variable */

		ColorTransferFunction *__pTransferFunction = nullptr;
		cudaArray_t __pTransferTableDevBufferArr[4] = { nullptr, nullptr, nullptr, nullptr };

		Light __lights[3];
		float __imgBasedSamplingStep;
		float __objectBasedSamplingStep;
		float __shininess = 40.f;

		const VolumeNumericMeta& __volumeNumericMeta;
		const bool &__volumeLoaded;
	};

	class ImageProcessor
	{
	private:
		friend RenderingEngine;

		/* member variable */
		const VolumeNumericMeta& __volumeNumericMeta;
		const bool &__volumeLoaded;

		MonoTransferFunction *__pTransferFunction = nullptr;
		cudaArray_t __transferTableBuffer = nullptr;

		SlicingPointManager __slicingPointMgr;
		AnchorManager __anchorMgr;

		float __samplingStepArr[3];

		/* constructor */
		ImageProcessor(const VolumeNumericMeta& volumeNumericMeta, const bool &volumeLoadedFlag);

		/* member function */
		void __init();
		void __release();
		void __syncTransferFunction();

	public:
		/* member function */
		Index2D<> getSlicingPointForScreen(const Size2D<> &screenSize, const SliceAxis axis);
		const Point3D &getSlicingPoint() const;

		void setSlicingPointFromScreen(const Size2D<> &screenSize, const Index2D<> &screenIdx, const SliceAxis axis);
		void setSlicingPointAdj(const Point3D &adj);
		void setSlicingPoint(const Point3D &point);

		const Point2D &getAnchorAdj(const SliceAxis axis) const;
		void setAnchorAdj(const float adjX, const float adjY, const SliceAxis axis);
		void adjustSlicingPoint(const float delta, const SliceAxis axis);
		void adjustAnchor(const float deltaX, const float deltaY, const SliceAxis axis);
		
		void adjustSamplingStep(const float delta, const SliceAxis axis);

		bool initTransferFunction();

		template <typename T>
		bool setTransferFunction(const T* const pData);

		template <typename T>
		std::vector<T> getTransferFunctionAs() const;

		void render(
			Pixel *const pScreen, const int screenWidth, const int screenHeight, const SliceAxis axis);

		~ImageProcessor();
	};

	/* member function */
	void loadVolume(const VolumeData &volumeData);
	bool isVolumeLoaded() const;

	const VolumeMeta &getVolumeMeta() const;

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

	bool __volumeLoaded = false;
};

template <typename T>
std::vector<T> RenderingEngine::VolumeRenderer::getTransferFunctionAs(
	const ColorChannelType funcType) const
{
	const int ITER = __pTransferFunction->PRECISION;
	std::vector<T> retVal;

	retVal.resize(ITER);

	const float *const pTransferTable = __pTransferFunction->getBuffer(funcType);

	for (int i = 0; i < ITER; i++)
		retVal[i] = static_cast<T>(pTransferTable[i]);

	return retVal;
}

template <typename T>
bool RenderingEngine::VolumeRenderer::setTransferFunction(const ColorChannelType colorType, const T* const pData)
{
	IF_F_RET_F(__pTransferFunction);

	__pTransferFunction->set(colorType, pData);
	__syncTransferFunction(colorType);

	SystemIndirectAccessor::getEventBroadcaster().notifyUpdateVolumeTransferFunction(colorType);

	return true;
}

template <typename T>
bool RenderingEngine::ImageProcessor::setTransferFunction(const T* const pData)
{
	IF_F_RET_F(__pTransferFunction);

	__pTransferFunction->set(pData);
	__syncTransferFunction();

	SystemIndirectAccessor::getEventBroadcaster().notifySliceTransferFunctionUpdate();

	return true;
}

template <typename T>
std::vector<T> RenderingEngine::ImageProcessor::getTransferFunctionAs() const
{
	const int ITER = __pTransferFunction->PRECISION;
	std::vector<T> retVal;

	retVal.resize(ITER);

	const float *pTransferTable = __pTransferFunction->getBuffer();

	for (int i = 0; i < ITER; i++)
		retVal[i] = static_cast<T>(pTransferTable[i]);

	return retVal;
}