#pragma once

#include <map>
#include <vector>
#include "tstring.h"

class MetaImageHeader
{
private:
	bool __loaded = false;
	std::map<std::tstring, std::vector<std::tstring>> __metaMap;

public:
	MetaImageHeader() = default;
	MetaImageHeader(const std::tstring &path);

	bool load(const std::tstring &path);
	bool isLoaded() const;

	const std::tstring &getValue(const std::tstring &name, const int idx = 0) const;

	template <typename T>
	T getValueAs(const std::tstring &name, const int idx = 0) const;
};

template <typename T>
T MetaImageHeader::getValueAs(const std::tstring &name, const int idx) const
{
	static_assert(std::is_arithmetic_v<T>, "T must be arithmetical type");

	return static_cast<T>(_ttof(__metaMap.at(name)[idx].c_str()));
}
