#pragma once

class SystemInitListener
{
public:
	/// <summary>
	/// 시스템이 초기화 될 때 호출되는 콜백
	/// </summary>
	virtual void onSystemInit() = 0;
};