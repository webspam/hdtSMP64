#include "stdafx.h"
#include "FrameworkImpl.h"
#include "StringImpl.h"
#include <debugapi.h>

#include "skse64_common/Relocation.h"
#include <skse64/skse64_common/skse_version.h>
#include "hdtSSEPhysics/Offsets.h"

namespace hdt
{
	FrameworkImpl::FrameworkImpl()
	{
	}

	FrameworkImpl::~FrameworkImpl()
	{
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
	
	float FrameworkImpl::getFrameInterval(bool raceMenu)
	{
		if (raceMenu)
			return *(float*)(RelocationManager::s_baseAddr + offset::GameStepTimer_NoSlowTime); // updateTimer instance + 0x1C
		else return *(float*)(RelocationManager::s_baseAddr + offset::GameStepTimer_SlowTime); // updateTimer instance + 0x18
	}

	FrameworkImpl * FrameworkImpl::instance()
	{
		static FrameworkImpl s;
		return &s;
	}

}
