#include "RenderingEngine.h"
#include "System.h"
#include "Debugger.h"

RenderingEngine RenderingEngine::__instance;

using namespace std;

namespace Device
{
	// static texture<ushort, 3, cudaReadModeNormalizedFloat> __ushortTex;
}

////////////////////////////////
//// RENDERING ENGINE START ////
////////////////////////////////

RenderingEngine::RenderingEngine() :
	volumeRenderer(__volumeMeta), imageProcessor(__volumeMeta)
{}

void RenderingEngine::loadVolume(const VolumeData &volumeData)
{
	__volumeMeta = volumeData.meta;

	// some initializing logic with cuda texture..

	volumeRenderer.__onLoadVolume();
	imageProcessor.__onLoadVolume();
}

void RenderingEngine::onLoadVolume(const VolumeData &volumeData)
{
	loadVolume(volumeData);
}

RenderingEngine &RenderingEngine::getInstance()
{
	return __instance;
}


///////////////////////////////
//// VOLUME RENDERER START ////
///////////////////////////////

RenderingEngine::VolumeRenderer::VolumeRenderer(const VolumeMeta &volumeMeta) :
	__volumeMeta(volumeMeta)
{

}

void RenderingEngine::VolumeRenderer::__onLoadVolume()
{
	// 새로운 볼륨이 렌더링 엔진에 적재된 후 나중에 호출되는 콜백 함수
	// 새로운 볼륨 적재 시 필요한 처리 루틴 작성
}

void RenderingEngine::VolumeRenderer::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight)
{
	
}

///////////////////////////////
//// IMAGE PROCESSOR START ////
///////////////////////////////

RenderingEngine::ImageProcessor::ImageProcessor(const VolumeMeta &volumeMeta) :
	__volumeMeta(volumeMeta)
{

}

void RenderingEngine::ImageProcessor::__onLoadVolume()
{

}

void RenderingEngine::ImageProcessor::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight)
{

}