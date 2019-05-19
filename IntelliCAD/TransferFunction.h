#pragma once

#include "Range.hpp"
#include "TypeEx.h"

class TransferFunction 
{
public:
	/* constructor */
	/// <summary>
	/// 생성자
	/// </summary>
	/// <param name="precision">
	/// 복셀 정밀도
	/// </param>
	explicit TransferFunction(const ushort precision);

	/* destructor */
	/// <summary>
	/// 소멸자
	/// </summary>
	~TransferFunction();

	/* member function */
	/// <summary>
	/// red transfer function에 필터를 적용한다.
	/// </summary>
	/// <param name="filter">
	/// 필터 전이구간
	/// </param>
	void setRed(const Range<ushort>& filter);

	/// <summary>
	/// green transfer function에 필터를 적용한다.
	/// </summary>
	/// <param name="filter">
	/// 필터 전이구간
	/// </param>
	void setGreen(const Range<ushort>& filter);

	/// <summary>
	/// blue transfer function에 필터를 적용한다.
	/// </summary>
	/// <param name="filter">
	/// 필터 전이구간
	/// </param>
	void setBlue(const Range<ushort>& filter);

	/// <summary>
	/// alpha transfer function에 필터를 적용한다.
	/// </summary>
	/// <param name="filter">
	/// 필터 전이구간
	/// </param>
	void setAlpha(const Range<ushort>& filter);

	/// <summary>
	/// red transfer function의 포인터를 가져온다.
	/// </summary>
	/// <returns>
	/// red transfer function 포인터
	/// </returns>
	const float* getRed() const;

	/// <summary>
	/// green transfer function의 포인터를 가져온다.
	/// </summary>
	/// <returns>
	/// green transfer function 포인터
	/// </returns>
	const float* getGreen() const;

	/// <summary>
	/// blue transfer function의 포인터를 가져온다.
	/// </summary>
	/// <returns>
	/// blue transfer function 포인터
	/// </returns>
	const float* getBlue() const;

	/// <summary>
	/// alpha transfer function의 포인터를 가져온다.
	/// </summary>
	/// <returns>
	/// alpha transfer function 포인터
	/// </returns>
	const float* getAlpha() const;

	/* member variable */
	/// <summary>
	/// 복셀 정밀도
	/// </summary>
	const ushort PRECISION;

private:
	/* member function */
	/// <summary>
	/// 메모리를 할당한다.
	/// </summary>
	void __malloc();

	/// <summary>
	/// transfer function 값을 계산한다.
	/// </summary>
	/// <param name="pTransferFunc">
	/// transfer function 포인터
	/// </param>
	/// <param name="filter">
	/// 필터 전이구간
	/// </param>
	void __calcTransferFunc(float* const pTransferFunc, const Range<ushort>& filter);

	/// <summary>
	/// 메모리를 해제한다.
	/// </summary>
	/// <param name="pTransferFunc">
	/// transfer function 포인터의 레퍼런스
	/// </param>
	void __free(float*& pTransferFunc);

	/* member variable */
	/// <summary>
	/// red transfer function
	/// </summary>
	float* __pRed = nullptr;

	/// <summary>
	/// green transfer function
	/// </summary>
	float* __pGreen = nullptr;

	/// <summary>
	/// blue transfer function
	/// </summary>
	float* __pBlue = nullptr;
	
	/// <summary>
	/// alpha transfer function
	/// </summary>
	float* __pAlpha = nullptr;
};