#pragma once

#include <vector>
#include "TypeEx.h"
#include "ObjectStream.h"
#include "ElementMeta.h"

class Serializable
{
private:
	friend ObjectStream;

public:

	virtual std::vector<ElementMeta> _getStreamMeta() const = 0;

	int serialAccess(int offset, ubyte *pBuffer, int bufferSize = 1) const;

	ObjectStream getStream() const;
};