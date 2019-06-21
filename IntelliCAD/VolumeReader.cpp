#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <filesystem>
#include "VolumeReader.h"
#include "MetaImageHeader.h"

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

		if (fin)
		{
			retVal.meta.fileName = filesystem::path(path).filename();
			retVal.meta.numeric.memSize.set(width, height, depth);
			retVal.meta.numeric.voxelPrecision = voxelPrecision;

			const int TOTAL_SIZE = (width * height * depth);
			retVal.pBuffer = shared_ptr<ushort[]>(new ushort[TOTAL_SIZE]);

			fin.read(reinterpret_cast<char *>(retVal.pBuffer.get()), TOTAL_SIZE * sizeof(ushort));
			fin.close();
		}

		return retVal;
	}

	VolumeData readMetaImage(const tstring &path)
	{
		VolumeData retVal;
		MetaImageHeader header;

		if (header.load(path))
		{
			retVal.meta.fileName = header.getValue(_T("ElementDataFile"));
			
			VolumeNumericMeta &numericMeta = retVal.meta.numeric;

			numericMeta.memSize.set(
				header.getValueAs<int>(_T("DimSize"), 0),
				header.getValueAs<int>(_T("DimSize"), 1),
				header.getValueAs<int>(_T("DimSize"), 2));

			numericMeta.spacing.set(
				header.getValueAs<float>(_T("ElementSpacing"), 0),
				header.getValueAs<float>(_T("ElementSpacing"), 1),
				header.getValueAs<float>(_T("ElementSpacing"), 2)
			);

			numericMeta.volSize.set(
				numericMeta.memSize.width * numericMeta.spacing.width,
				numericMeta.memSize.height * numericMeta.spacing.height,
				numericMeta.memSize.depth * numericMeta.spacing.depth
			);

			const tstring RAW_PATH = (filesystem::path(path).parent_path() /= retVal.meta.fileName);
			ifstream fin(RAW_PATH, ifstream::binary);

			if (fin)
			{
				const int TOTAL_SIZE = numericMeta.memSize.getTotalSize();
				retVal.pBuffer = shared_ptr<ushort[]>(new ushort[TOTAL_SIZE]);

				fin.read(reinterpret_cast<char *>(retVal.pBuffer.get()), TOTAL_SIZE * sizeof(ushort));
				fin.close();

				short *const pBuffer = reinterpret_cast<short *>(retVal.pBuffer.get());

				short minVal = SHRT_MAX;
				short maxVal = SHRT_MIN;

				#pragma omp parallel
				{
					short threadMin = SHRT_MAX;
					short threadMax = SHRT_MIN;

					#pragma omp for
					for (int i = 0; i < TOTAL_SIZE; i++)
					{
						threadMin = min(threadMin, pBuffer[i]);
						threadMax = max(threadMax, pBuffer[i]);
					}

					#pragma omp critical(reduction_min)
					minVal = min(minVal, threadMin);

					#pragma omp critical(reduction_max)
					maxVal = max(maxVal, threadMax);

					#pragma omp for
					for (int i = 0; i < TOTAL_SIZE; i++)
						pBuffer[i] -= minVal;
				}

				numericMeta.voxelPrecision = static_cast<ushort>(maxVal - minVal);

				// Padding
				if (numericMeta.voxelPrecision < USHRT_MAX)
					numericMeta.voxelPrecision++;
			}
		}

		return retVal;
	}
}