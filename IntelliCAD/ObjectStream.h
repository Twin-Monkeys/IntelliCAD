#pragma once

#include "ReadStream.hpp"

class ObjectStream : public ReadStream
{
private:
	int __cursor = 0;
	const class Serializable *const pObject;

public:
	explicit ObjectStream(const class Serializable *pObject);

	virtual bool get(ubyte &buffer) override;
	virtual int get(void *pBuffer, int bufferSize) override;

	virtual int getStreamSize() const override;
};
