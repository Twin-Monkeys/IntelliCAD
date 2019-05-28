#include "ElementMeta.h"

ElementMeta::ElementMeta(const int elemSize, const void *const pElemPtr) :
	elemSize(elemSize), pElemPtr(reinterpret_cast<const ubyte *>(pElemPtr))
{}