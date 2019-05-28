#pragma once

#include "TypeEx.h"
#include "StreamAnchorType.h"

class ReadStream
{
public:
	virtual bool get(ubyte &buffer) = 0;
	virtual int get(void *pBuffer, int bufferSize) = 0;

	template <typename T>
	bool getAs(T &buffer);

	virtual int getStreamSize() const = 0;
};

template <typename T>
bool ReadStream::getAs(T &buffer)
{
	const int TYPE_SIZE = sizeof(T);
	const int COUNT = get(&buffer, TYPE_SIZE);

	if (COUNT != TYPE_SIZE)
		return false;

	return true;
}