#pragma once

#include <memory>
#include <fstream>
#include "tstring.h"
#include "GPUVolume.h"

namespace VolumeReader
{
	template <typename T>
	bool readDen(const std::tstring &path, GPUVolume &volume, const int bufferSize = 0xA00000);

	template <typename T>
	std::shared_ptr<GPUVolume> readDen(
		const std::tstring &path, const Size3D<> &size, const int elementSize, const int readBufferSize = 0xA00000);

	template <typename T>
	std::shared_ptr<GPUVolume> readDen(
		const std::tstring &path, const Size3D<> &size,
		const int elementSize, const int elementPrecision, const int readBufferSize = 0xA00000);

	template <typename T>
	bool readDen(const std::tstring &path, GPUVolume &volume, const int readBufferSize)
	{
		using namespace std;

		ifstream fin(path, ifstream::binary);

		if (!fin)
			return false;

		ubyte *const pBuffer = new ubyte[readBufferSize];
		T *const pDevPtr = volume.getDevBuffer();

		const int TOTAL_SIZE = volume.SIZE.getTotalSize<T>();
		int readSize = 0;

		do
		{
			fin.read(reinterpret_cast<char *>(pBuffer), readBufferSize);
			const int count = fin.gcount();

			if (!count)
				break;

			cudaMemcpy(pDevPtr + readSize, pBuffer, count, cudaMemcpyKind::cudaMemcpyHostToDevice);

			readSize += count;
		}
		while (TOTAL_SIZE > readSize);

		delete[] pBuffer;

		return true;
	}

	template <typename T>
	std::shared_ptr<GPUVolume> readDen(
		const std::tstring &path, const Size3D<> &size, const int elementSize, const int bufferSize)
	{
		using namespace std;

		shared_ptr<GPUVolume> pRetVal = make_shared<GPUVolume>(size, elementSize);

		if (!readDen(path, *pRetVal, bufferSize))
			return nullptr;

		return pRetVal;
	}

	template <typename T>
	std::shared_ptr<GPUVolume> readDen(
		const std::tstring &path, const Size3D<> &size,
		const int elementSize, const int elementPrecision, const int readBufferSize)
	{
		using namespace std;

		shared_ptr<GPUVolume> pRetVal = make_shared<GPUVolume>(size, elementSize, elementPrecision);

		if (!readDen(path, *pRetVal, bufferSize))
			return nullptr;

		return pRetVal;
	}
}
