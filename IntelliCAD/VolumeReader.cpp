#include <fstream>
#include "VolumeReader.h"

using namespace std;

namespace VolumeReader
{
	VolumeData readDen(
		const tstring &path, const int width, const int height, const int depth)
	{
		return readDen(path, width, height, depth, USHRT_MAX);
	}

	VolumeData readDen(
		const tstring &path,
		const int width, const int height, const int depth, const ushort voxelPrecision)
	{
		VolumeData retVal;
		
		ifstream fin(path, ifstream::binary);

		if (!fin)
			return retVal;

		const int TOTAL_SIZE = (width * height * depth);

		retVal.pBuffer = shared_ptr<ushort[]>(new ushort[TOTAL_SIZE]);
		fin.read(reinterpret_cast<char *>(retVal.pBuffer.get()), TOTAL_SIZE * sizeof(ushort));

		retVal.meta.size.set(width, height, depth);
		retVal.meta.voxelPrecision = voxelPrecision;

		return retVal;
	}
}