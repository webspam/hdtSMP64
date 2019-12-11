#include "stdafx.h"
#include "HookArmor.h"
#include "HookUtils.h"
#include "Offsets.h"
#include "../hdtSSEUtils/NetImmerseUtils.h"
#include "FrameworkImpl.h"

#include <common\IPrefix.h>

#include <skse64\skse64\GameData.h>
#include <skse64\skse64\NiNodes.h>

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
			FrameworkImpl::instance()->getArmorAttachEventDispatcher()->dispatch(event);

			auto ret = CALL_MEMBER_FN(this, unk001CB0E0)(armor, skeleton, unk3, unk4, unk5, unk6);

			event.attachedNode = ret;
			event.hasAttached = true;
			FrameworkImpl::instance()->getArmorAttachEventDispatcher()->dispatch(event);

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
}