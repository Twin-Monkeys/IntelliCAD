#pragma once

#include "Range.hpp"
#include "TypeEx.h"

class TransferFunction 
{
public:
	/* constructor */
	/// <summary>
	/// ������
	/// </summary>
	/// <param name="precision">
	/// ���� ���е�
	/// </param>
	explicit TransferFunction(const ushort precision);

	/* destructor */
	/// <summary>
	/// �Ҹ���
	/// </summary>
	~TransferFunction();

	/* member function */
	/// <summary>
	/// red transfer function�� ���͸� �����Ѵ�.
	/// </summary>
	/// <param name="filter">
	/// ���� ���̱���
	/// </param>
	void setRed(const Range<ushort>& filter);

	/// <summary>
	/// green transfer function�� ���͸� �����Ѵ�.
	/// </summary>
	/// <param name="filter">
	/// ���� ���̱���
	/// </param>
	void setGreen(const Range<ushort>& filter);

	/// <summary>
	/// blue transfer function�� ���͸� �����Ѵ�.
	/// </summary>
	/// <param name="filter">
	/// ���� ���̱���
	/// </param>
	void setBlue(const Range<ushort>& filter);

	/// <summary>
	/// alpha transfer function�� ���͸� �����Ѵ�.
	/// </summary>
	/// <param name="filter">
	/// ���� ���̱���
	/// </param>
	void setAlpha(const Range<ushort>& filter);

	/// <summary>
	/// red transfer function�� �����͸� �����´�.
	/// </summary>
	/// <returns>
	/// red transfer function ������
	/// </returns>
	const float* getRed() const;

	/// <summary>
	/// green transfer function�� �����͸� �����´�.
	/// </summary>
	/// <returns>
	/// green transfer function ������
	/// </returns>
	const float* getGreen() const;

	/// <summary>
	/// blue transfer function�� �����͸� �����´�.
	/// </summary>
	/// <returns>
	/// blue transfer function ������
	/// </returns>
	const float* getBlue() const;

	/// <summary>
	/// alpha transfer function�� �����͸� �����´�.
	/// </summary>
	/// <returns>
	/// alpha transfer function ������
	/// </returns>
	const float* getAlpha() const;

	/* member variable */
	/// <summary>
	/// ���� ���е�
	/// </summary>
	const ushort PRECISION;

private:
	/* member function */
	/// <summary>
	/// �޸𸮸� �Ҵ��Ѵ�.
	/// </summary>
	void __malloc();

	/// <summary>
	/// transfer function ���� ����Ѵ�.
	/// </summary>
	/// <param name="pTransferFunc">
	/// transfer function ������
	/// </param>
	/// <param name="filter">
	/// ���� ���̱���
	/// </param>
	void __calcTransferFunc(float* const pTransferFunc, const Range<ushort>& filter);

	/// <summary>
	/// �޸𸮸� �����Ѵ�.
	/// </summary>
	/// <param name="pTransferFunc">
	/// transfer function �������� ���۷���
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