#pragma once

#include "IFramework.h"
#include <unordered_map>

namespace hdt
{
	class FrameworkImpl : public IFramework
	{
	public:
		FrameworkImpl();
		~FrameworkImpl();

		static FrameworkImpl* instance();
		virtual APIVersion getApiVersion() override;

		virtual bool isSupportedSkyrimVersion(uint32_t version) override;

		virtual float getFrameInterval(bool raceMenu) override;
		
	protected:

		bool m_isHooked = false;
	};
}
