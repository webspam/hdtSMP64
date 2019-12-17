#include "HookEvents.h"
#include "EventDispatcherImpl.h"

namespace hdt {
	EventDispatcherImpl<FrameEvent>			g_frameEventDispatcher;
	EventDispatcherImpl<ShutdownEvent>		g_shutdownEventDispatcher;
	EventDispatcherImpl<ArmorAttachEvent>	g_armorAttachEventDispatcher;
}
