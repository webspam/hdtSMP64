#pragma once

#include "skse64/NiObjects.h"

#include "EventDispatcherImpl.h"

namespace hdt
{
	struct ArmorAttachEvent
	{
		NiNode*		armorModel = nullptr;
		NiNode*		skeleton = nullptr;
		NiAVObject* attachedNode = nullptr;
		bool		hasAttached = false;
	};
	
	struct FrameEvent
	{
		bool frameEnd;
	};

	struct ShutdownEvent
	{
	};

	extern EventDispatcherImpl<FrameEvent>			g_frameEventDispatcher;
	extern EventDispatcherImpl<ShutdownEvent>		g_shutdownEventDispatcher;
	extern EventDispatcherImpl<ArmorAttachEvent>	g_armorAttachEventDispatcher;
}
