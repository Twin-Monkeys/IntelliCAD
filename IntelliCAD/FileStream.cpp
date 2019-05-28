#include "FileStream.h"

using namespace std;

FileStream::FileStream(const string &path) :
	fin(path, ifstream::binary)
{}

bool FileStream::open(const string &path)
{
	fin.open(path, ifstream::binary);

	return fin.is_open();
}

bool FileStream::isOpened() const
{
	return fin.is_open();
}

bool FileStream::get(ubyte &buffer)
{
	fin.read(reinterpret_cast<char *>(&buffer), 1);
	return fin.gcount();
}

int FileStream::get(void *const pBuffer, const int bufferSize)
{
	fin.read(reinterpret_cast<char *>(pBuffer), bufferSize);
	return static_cast<int>(fin.gcount());
}



int FileStream::getStreamSize() const
{
	const streampos CUR_CURSOR = fin.tellg();

	fin.seekg(0, ifstream::end);
	const streampos RET_VAL = fin.tellg();

	fin.seekg(CUR_CURSOR);
	return static_cast<int>(RET_VAL);
}

FileStream::operator bool() const
{
	return fin.is_open();
}

bool FileStream::operator!() const
{
	return !fin;
}
