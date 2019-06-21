#pragma once

#include <fstream>
#include "tchar.h"

namespace std
{
	using tfstream = basic_fstream<TCHAR>;
	using tofstream = basic_ofstream<TCHAR>;
	using tifstream = basic_ifstream<TCHAR>;
}