#include "skse64/GameReferences.h"

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
		_DMESSAGE("received ArmorAttachEvent(armorModel=%016llX, skeleton=%016llX, attachedNode=%016llX, hasAttached=%s)", (uintptr_t)e.armorModel, (uintptr_t)e.skeleton, (uintptr_t)e.attachedNode, (uintptr_t)e.hasAttached ? "true" : "false");
	}
}
