#include <detours.h>

#include "skse64/NiObjects.h"

#include "Hooks.h"
#include "HookEvents.h"
#include "Offsets.h"

namespace hdt
{
	struct Unk001CB0E0
	{
		MEMBER_FN_PREFIX(Unk001CB0E0);

		DEFINE_MEMBER_FN_HOOK(unk001CB0E0, NiAVObject*, offset::ArmorAttachFunction, NiNode* armor, NiNode* skeleton, void* unk3, char unk4, char unk5, void* unk6);
		NiAVObject* unk001CB0E0(NiNode* armor, NiNode* skeleton, void* unk3, char unk4, char unk5, void* unk6)
		{
			ArmorAttachEvent event;
			event.armorModel = armor;
			event.skeleton = skeleton;
			g_armorAttachEventDispatcher.dispatch(event);

			auto ret = CALL_MEMBER_FN(this, unk001CB0E0)(armor, skeleton, unk3, unk4, unk5, unk6);

			event.attachedNode = ret;
			event.hasAttached = true;
			g_armorAttachEventDispatcher.dispatch(event);

			return ret;
		}
	};

	void hookArmor()
	{
		DetourAttach((void**)Unk001CB0E0::_unk001CB0E0_GetPtrAddr(), (void*)GetFnAddr(&Unk001CB0E0::unk001CB0E0));
	}

	void unhookArmor()
	{
		DetourDetach((void**)Unk001CB0E0::_unk001CB0E0_GetPtrAddr(), (void*)GetFnAddr(&Unk001CB0E0::unk001CB0E0));
	}

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
		g_frameEventDispatcher.dispatch(e);
		CALL_MEMBER_FN(this, onFrame)();
		e.frameEnd = true;
		g_frameEventDispatcher.dispatch(e);
	}

	auto oldShutdown = (void (*)(bool))(RelocationManager::s_baseAddr + offset::GameShutdownFunction);
	void shutdown(bool arg0)
	{
		g_shutdownEventDispatcher.dispatch(ShutdownEvent());
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

	void hookAll()
	{
		DetourRestoreAfterWith();
		DetourTransactionBegin();
		hookEngine();
		hookArmor();
		DetourTransactionCommit();
	}

	void unhookAll()
	{
		DetourRestoreAfterWith();
		DetourTransactionBegin();
		unhookEngine();
		unhookArmor();
		DetourTransactionCommit();
	}
}
