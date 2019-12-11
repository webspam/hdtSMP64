#include "stdafx.h"
#include "HookEngine.h"
#include "IFramework.h"
#include "HookUtils.h"
#include "FrameworkImpl.h"
#include "Offsets.h"

namespace hdt
{
	struct UnkEngine
	{
		MEMBER_FN_PREFIX(UnkEngine);

		DEFINE_MEMBER_FN_HOOK(onFrame, void, offset::GameLoopFunction);

		void onFrame();
	};

	void UnkEngine::onFrame()
	{
		FrameEvent e;
		e.frameEnd = false;
		FrameworkImpl::instance()->getFrameEventDispatcher()->dispatch(e);
		CALL_MEMBER_FN(this, onFrame)();
		e.frameEnd = true;
		FrameworkImpl::instance()->getFrameEventDispatcher()->dispatch(e);
	}

	auto oldShutdown = (void (*)(bool))(hookGetBaseAddr() + offset::GameShutdownFunction);
	void shutdown(bool arg0)
	{
		FrameworkImpl::instance()->getShutdownEventDispatcher()->dispatch(ShutdownEvent());
		oldShutdown(arg0);
	}
		
	void hookEngine()
	{
		DetourAttach((void**)UnkEngine::_onFrame_GetPtrAddr(), (void*)GetFnAddr(&UnkEngine::onFrame));
		DetourAttach((void**)&oldShutdown, (void*)shutdown);
	}

	void unhookEngine()
	{
		DetourDetach((void**)UnkEngine::_onFrame_GetPtrAddr(), (void*)GetFnAddr(&UnkEngine::onFrame));
		DetourDetach((void**)&oldShutdown, (void*)shutdown);
	}

}
