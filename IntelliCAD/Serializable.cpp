#include "Serializable.h"
#include "NumberUtility.hpp"

using namespace std;

int Serializable::serialAccess(const int offset, ubyte *const pBuffer, const int bufferSize) const
{
	if (offset < 0)
		return 0;

	int bufferOffset = 0;
	int accumedOffset = 0;
	int streamOffset = offset;
	int bufferRemaining = bufferSize;

	vector<ElementMeta> elemInfos = _getStreamMeta();
	for (const ElementMeta &info : elemInfos)
	{
		if (streamOffset < (accumedOffset + info.elemSize))
		{
			int localOffset = (streamOffset - accumedOffset);
			const int COPY_SIZE = (info.elemSize - localOffset);

			if (bufferRemaining < COPY_SIZE)
			{
				memcpy(pBuffer + bufferOffset, info.pElemPtr + localOffset, bufferRemaining);
				return bufferSize;
			}

			memcpy(pBuffer + bufferOffset, info.pElemPtr + localOffset, COPY_SIZE);

			bufferOffset += COPY_SIZE;
			streamOffset += COPY_SIZE;
			bufferRemaining -= COPY_SIZE;

			if (!bufferRemaining)
				return bufferSize;
		}

		accumedOffset += info.elemSize;
	}

	return bufferOffset;
}

ObjectStream Serializable::getStream() const
{
	return ObjectStream(this);
}