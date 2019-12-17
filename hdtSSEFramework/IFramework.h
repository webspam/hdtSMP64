#pragma once


namespace hdt
{
	class IFramework
	{
	public:

		union APIVersion
		{
		public:
			APIVersion(uint32_t major, uint32_t minor) : majorVersion(major), minorVersion(minor) {}
			struct
			{
				uint32_t minorVersion;
				uint32_t majorVersion;
			};
			uint64_t version;
		};

		static IFramework* instance();

		virtual APIVersion getApiVersion() = 0;
		virtual bool isSupportedSkyrimVersion(uint32_t version) = 0;
		
		virtual float getFrameInterval(bool raceMenu) = 0;

	};
}

extern "C"
{
	hdt::IFramework* hdtGetFramework();
}
