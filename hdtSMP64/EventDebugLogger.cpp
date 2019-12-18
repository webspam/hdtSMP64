#include "skse64/GameReferences.h"
#include "skse64/NiObjects.h"
#include "skse64/NiNodes.h"

#include "EventDebugLogger.h"

namespace hdt
{
	EventResult EventDebugLogger::ReceiveEvent(TESCellAttachDetachEvent* evn,
		EventDispatcher<TESCellAttachDetachEvent>* dispatcher)
	{
		if (evn && evn->reference && evn->reference->formType == Character::kTypeID)
		{
			_DMESSAGE("received TESCellAttachDetachEvent(formID %08llX, name %s, attached=%s)", evn->reference->formID, evn->reference->baseForm->GetFullName(), evn->attached ? "true" : "false");
		}
		return kEvent_Continue;
	}

	EventResult EventDebugLogger::ReceiveEvent(TESMoveAttachDetachEvent* evn,
		EventDispatcher<TESMoveAttachDetachEvent>* dispatcher)
	{
		if (evn && evn->reference && evn->reference->formType == Character::kTypeID)
		{
			_DMESSAGE("received TESMoveAttachDetachEvent(formID %08llX, name %s, attached=%s)", evn->reference->formID, evn->reference->baseForm->GetFullName(), evn->attached ? "true" : "false");
		}
		return kEvent_Continue;
	}

	void EventDebugLogger::onEvent(const ArmorAttachEvent& e)
	{
		_DMESSAGE("received ArmorAttachEvent(armorModel=%s (%016llX), skeleton=%s (%016llX), attachedNode=%s (%016llX), hasAttached=%s)", 
			e.armorModel ? e.armorModel->m_name : "null", 
			(uintptr_t)e.armorModel, 
			e.skeleton ? e.skeleton->m_name : "null",
			(uintptr_t)e.skeleton, 
			e.attachedNode ? e.attachedNode->m_name : "null", 
			(uintptr_t)e.attachedNode, 
			(uintptr_t)e.hasAttached ? "true" : "false");
	}
}
