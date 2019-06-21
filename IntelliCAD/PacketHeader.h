#pragma once

#include "tstring.h"

enum class ProtocolType
{
	
	CONNECTION_CHECK,
	CONNECTION_RESPONSE,
	OBJECT,
	FILE_REQUEST,
	FILE_RESPONSE,
	PROTOCOL_SUCCESS, // 마지막을 알림
	LOGIN,
	OK,
	NOK,
	DB_ERROR,

	MAX
};

enum class ObjectType
{
	
	MSG,

	INT,

	//class
	USER_INFO, 
	DUMMY_CLASS,
	
	//container
	VECTOR_UINT32T,
	VECTOR_STRING,

	//file
	FILE_TXT,
	FILE_IMAGE,
	
	MAX
};

class PacketHeader
{
private:

	const ProtocolType __protocolType;
	
	const ObjectType __objectType;
	
	const uint32_t __byteCount;
	

public:
	
	PacketHeader(const ProtocolType protocolType, const ObjectType objectType, const uint32_t byteCount);

	ProtocolType getProtocolType() const; 
	ObjectType getObjectType() const;
	uint32_t getByteCount() const;
};