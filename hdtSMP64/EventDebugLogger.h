#pragma once

#include "skse64/GameEvents.h"

#include "HookEvents.h"
#include "IEventListener.h"

namespace hdt
{
	class EventDebugLogger 
		: public IEventListener<ArmorAttachEvent>
		, public BSTEventSink<TESCellAttachDetachEvent>
		, public BSTEventSink<TESMoveAttachDetachEvent>
	{
	protected:
		virtual EventResult ReceiveEvent(TESCellAttachDetachEvent* evn, EventDispatcher<TESCellAttachDetachEvent>* dispatcher) override;
		virtual EventResult ReceiveEvent(TESMoveAttachDetachEvent* evn, EventDispatcher<TESMoveAttachDetachEvent>* dispatcher) override;

		virtual void onEvent(const ArmorAttachEvent&) override;
	};
}
