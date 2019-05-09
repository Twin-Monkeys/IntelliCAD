#include "RenderingEngine.h"
#include "System.h"
#include "Debugger.h"

RenderingEngine::RenderingEngine()
{
	System::getSystemContents().
		getEventBroadcaster().addSetVolumeListener(*this);
}

void RenderingEngine::setVolume(const GPUVolume *const pVolume)
{

}

void RenderingEngine::render(Pixel *const pScreen, const int screenWidth, const int screenHeight)
{

}

void RenderingEngine::onSetVolume(const GPUVolume *const pVolume)
{
	setVolume(pVolume);
	Debugger::popMessageBox(_T("onSetVolume"));
}