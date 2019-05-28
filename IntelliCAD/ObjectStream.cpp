#include "ObjectStream.h"
#include "Serializable.h"

using namespace std;

ObjectStream::ObjectStream(const Serializable *const pObject) :
	pObject(pObject)
{}

bool ObjectStream::get(ubyte &buffer)
{
	const bool VAILD = pObject->serialAccess(__cursor, &buffer);
	if (VAILD)
		__cursor++;

	return VAILD;
}

int ObjectStream::get(void *const pBuffer, const int bufferSize)
{
	const int COUNT = pObject->serialAccess(__cursor, static_cast<ubyte *>(pBuffer), bufferSize);
	__cursor += COUNT;

	return COUNT;
}

int ObjectStream::getStreamSize() const
{
	int retVal = 0;

	for (const ElementMeta &info : pObject->_getStreamMeta())
		retVal += info.elemSize;

	return retVal;
}