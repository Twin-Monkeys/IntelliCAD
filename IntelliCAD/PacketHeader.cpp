#pragma once

#include "stdafx.h"
#include "PacketHeader.h"


PacketHeader::PacketHeader(const ProtocolType protocolType, const ObjectType objectType, const uint32_t byteCount)
	: __protocolType(protocolType), __objectType(objectType), __byteCount(byteCount)
{ 

}

ProtocolType PacketHeader::getProtocolType() const
{ 
	return __protocolType; 
}

ObjectType PacketHeader::getObjectType() const
{ 
	return __objectType; 
}

uint32_t PacketHeader::getByteCount() const
{ 
	return __byteCount; 
}