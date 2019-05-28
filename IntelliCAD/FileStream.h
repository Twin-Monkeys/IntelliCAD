#pragma once

#include <fstream>
#include <string>
#include "ReadStream.hpp"

class FileStream : public ReadStream
{
private:
	mutable std::ifstream fin;

public:
	FileStream() = default;
	explicit FileStream(const std::string &path);

	bool open(const std::string &path);
	bool isOpened() const;

	virtual bool get(ubyte &buffer) override;
	virtual int get(void *pBuffer, int bufferSize) override;

	virtual int getStreamSize() const override;

	operator bool() const;
	bool operator!() const;
};
