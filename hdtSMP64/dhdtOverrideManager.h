#pragma once
#include "DynamicHDT.h"
#include <fstream>

extern bool g_hasPapyrusExtension;

namespace hdt {
	namespace Override {

		//The formID of the armoraddon in ArmorAttachEvent cannot be acquired, which makes it impossible to check override by the formID upon attaching armoraddon.
		class OverrideManager{
		public:
			OverrideManager() = default;
			~OverrideManager() {};

			static OverrideManager* GetSingleton();

			void saveOverrideData(std::string save_name);

			void loadOverrideData(std::string save_name);

			std::string queryOverrideData();

			bool registerOverride(UInt32 actor_formID, std::string old_file_path, std::string new_file_path);

			std::string checkOverride(UInt32 actor_formID, std::string old_file_path);

		protected:
			std::unordered_map<UInt32, std::unordered_map<std::string, std::string>> m_ActorPhysicsFileSwapList;
		};
	}
}