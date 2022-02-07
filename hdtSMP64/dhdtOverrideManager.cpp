#include "dhdtOverrideManager.h"

using namespace hdt::Override;

bool g_hasPapyrusExtension = false;

OverrideManager* hdt::Override::OverrideManager::GetSingleton() 
{
	static OverrideManager g_overrideManager;
	return &g_overrideManager;
}

bool checkPapyrusExtension() {
	std::ofstream ifs("Data/Scripts/DynamicHDT.pex", std::ios::in | std::ios::_Nocreate);
	if (!ifs || !ifs.is_open()) {
		g_hasPapyrusExtension = false;
		return false;
	}
	ifs.close();
	g_hasPapyrusExtension = true;
	return true;
}

void hdt::Override::OverrideManager::saveOverrideData(std::string save_name)
{
	if (!g_hasPapyrusExtension)return;

	Console_Print("[DynamicHDT] -- Saving override data to \"%s\".", (OVERRIDE_SAVE_PATH + save_name + ".dhdt").c_str());
	std::ofstream ofs(OVERRIDE_SAVE_PATH + save_name + ".dhdt", std::ios::out);
	if (!ofs) {
		Console_Print("[DynamicHDT] Warning! -- Failed writing override file. File \"%s\" is not accessable.", (OVERRIDE_SAVE_PATH + save_name + ".dhdt").c_str());
		return;
	}
	for (auto& e : m_ActorPhysicsFileSwapList) {
		char buff[16];
		sprintf_s(buff, "%08X", e.first);
		ofs << std::hex << buff << " " <<std::dec<< e.second.size() << std::endl;
		for (auto& e1 : e.second) {
			if (e1.second.empty())continue;
			ofs << e1.first << "\t" << e1.second << std::endl;
		}
	}
	ofs.close();
}

void hdt::Override::OverrideManager::loadOverrideData(std::string save_name)
{

	if (!checkPapyrusExtension())return;

	save_name = save_name.substr(0, save_name.find_last_of("."));

	std::ifstream ifs(OVERRIDE_SAVE_PATH + save_name + ".dhdt", std::ios::in | std::ios::_Nocreate);
	if (!ifs) {
		Console_Print("[DynamicHDT] Warning! -- Failed reading override file. File \"%s\" doesn't exist or is not accessable.", (OVERRIDE_SAVE_PATH + save_name + ".dhdt").c_str());
		return;
	}
	try {
		while (!ifs.eof()) {
			UInt32 actor_formID, override_size = 0;
			ifs >> std::hex >> actor_formID >> override_size;
			for (int i = 0; i < override_size; ++i) {
				std::string orig_physics_file, override_physics_file;
				ifs >> orig_physics_file >> override_physics_file;
				this->registerOverride(actor_formID, orig_physics_file, override_physics_file);
			}
		}
	}catch (std::exception& e) {

			Console_Print("[DynamicHDT] ERROR! -- Failed parsing override data.");

			Console_Print("[DynamicHDT] Error(): %s\nWhat():\n\t%s", typeid(e).name(), e.what());

			return;
	}
	ifs.close();
}

std::string hdt::Override::OverrideManager::queryOverrideData()
{
	std::string console_print("[DynamicHDT] -- Querying existing override data...\n");
	
	for (auto i : m_ActorPhysicsFileSwapList) {
		console_print += "Actor formID: " + util::UInt32toString(i.first) + "\t" + std::to_string(i.second.size()) + "\n";
		for (auto j : i.second) {
			console_print += "\tOriginal file: " + j.first + "\n\t\t| Override: " + j.second + "\n";
		}
	}
	
	console_print += "[DynamicHDT] -- Query finished...\n";
	return console_print;
}

bool hdt::Override::OverrideManager::registerOverride(UInt32 actor_formID, std::string old_file_path, std::string new_file_path)
{
	if (old_file_path.empty())return false;
	for (auto& e : m_ActorPhysicsFileSwapList[actor_formID]) {
		if (e.second == old_file_path) {
			old_file_path = e.first;
		}
	}
	m_ActorPhysicsFileSwapList[actor_formID][old_file_path] = new_file_path;
	return true;
}

std::string hdt::Override::OverrideManager::checkOverride(UInt32 actor_formID, std::string old_file_path)
{
	auto iter1 = m_ActorPhysicsFileSwapList.find(actor_formID);
	if (iter1 != m_ActorPhysicsFileSwapList.end()) {
		auto iter2 = iter1->second.find(old_file_path);
		if (iter2 != iter1->second.end()) {
			return iter2->second;
		}
	}
	return std::string();
}
