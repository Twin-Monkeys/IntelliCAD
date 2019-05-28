#include "SystemIndirectAccessor.h"
#include "System.h"

namespace SystemIndirectAccessor
{
	EventBroadcaster &getEventBroadcaster()
	{
		return System::getSystemContents().getEventBroadcaster();
	}
}