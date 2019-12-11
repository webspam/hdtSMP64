#include "stdafx.h"
#include "FrameworkImpl.h"
#include "HookUtils.h"
#include "HookEngine.h"
#include "HookArmor.h"
#include "StringImpl.h"
#include "Offsets.h"
#include <debugapi.h>

#include <skse64/skse64_common/skse_version.h>

namespace hdt
{
	FrameworkImpl::FrameworkImpl()
	{
	}

	FrameworkImpl::~FrameworkImpl()
	{
		unhook();
	}

	IFramework::APIVersion FrameworkImpl::getApiVersion()
	{
		return APIVersion(1, 2);
	}

	bool FrameworkImpl::isSupportedSkyrimVersion(uint32_t version)
	{
		return version == CURRENT_RELEASE_RUNTIME;
	}

	IString * FrameworkImpl::getString(const char * strBegin, const char * strEnd)
	{
		if (!strBegin) return nullptr;
		if (!strEnd) strEnd = strBegin + strlen(strBegin);
		return StringManager::instance()->get(strBegin, strEnd);
	}

	IEventDispatcher<void*>* FrameworkImpl::getCustomEventDispatcher(IString * name)
	{
		std::lock_guard<decltype(m_customEventLock)> l(m_customEventLock);
		auto iter = m_customEventDispatchers.find(name);
		if (iter == m_customEventDispatchers.end())
		{
			auto dispatcher = std::unique_ptr<EventDispatcherImpl<void*>>(new EventDispatcherImpl<void*>);
			iter = m_customEventDispatchers.insert(std::make_pair(name, std::move(dispatcher))).first;
		}
		return iter->second.get();
	}

	IEventDispatcher<FrameEvent>* FrameworkImpl::getFrameEventDispatcher()
	{
		return &m_frameEventDispatcher;
	}

	IEventDispatcher<ShutdownEvent>* FrameworkImpl::getShutdownEventDispatcher()
	{
		return &m_shutdownEventDispatcher;
	}

	IEventDispatcher<ArmorAttachEvent>* FrameworkImpl::getArmorAttachEventDispatcher()
	{
		return &m_armorAttachEventDispatcher;
	}
	
	float FrameworkImpl::getFrameInterval(bool raceMenu)
	{
		if (raceMenu)
			return *(float*)(hookGetBaseAddr() + offset::GameStepTimer_NoSlowTime); // updateTimer instance + 0x1C
		else return *(float*)(hookGetBaseAddr() + offset::GameStepTimer_SlowTime); // updateTimer instance + 0x18
	}
	
	void FrameworkImpl::hook()
	{
		if (!m_isHooked)
		{
			DetourRestoreAfterWith();
			DetourTransactionBegin();
			hookEngine();
			hookArmor();
			
			DetourTransactionCommit();
			m_isHooked = true;
		}
	}

	void FrameworkImpl::unhook()
	{
		if (m_isHooked)
		{
			DetourRestoreAfterWith();
			DetourTransactionBegin();
			unhookEngine();
			unhookArmor();
			DetourTransactionCommit();
			m_isHooked = false;
		}
	}

	FrameworkImpl * FrameworkImpl::instance()
	{
		static FrameworkImpl s;
		return &s;
	}

}