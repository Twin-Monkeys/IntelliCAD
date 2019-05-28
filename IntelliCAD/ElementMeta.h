#pragma once

#include "TypeEx.h"

class ElementMeta
{
public:
	int elemSize;
	const ubyte *pElemPtr;

	ElementMeta(int elemSize, const void *pElemPtr);
};
