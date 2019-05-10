#include "VolumeReader.hpp"

using namespace std;

namespace VolumeReader
{
	bool readDen(
		const std::tstring &path, GPUVolume &volume,
		const int width, const int height, const int depth,
		const VoxelFormatType voxelFormat, const int voxelPrecision)
	{
		using namespace std;

		ifstream fin(path, ifstream::binary);

		if (!fin)
			return false;

		const int TOTAL_MEM_SIZE = (width * height * depth * (1 << voxelFormat));
		ubyte *const pBuffer = new ubyte[TOTAL_MEM_SIZE];

		volume.init(pBuffer, width, height, depth, voxelFormat, voxelPrecision);

		delete[] pBuffer;

		return true;
	}
}