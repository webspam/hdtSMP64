#include "stdafx.h"

#include "skse64/GameMenus.h"
#include "skse64/GameReferences.h"
#include "skse64/ObScript.h"
#include "skse64/PluginAPI.h"
#include "skse64_common/skse_version.h"
#include "skse64_common/SafeWrite.h"

#include "../hdtSSEUtils/LogUtils.h"

#include "ArmorManager.h"
#include "config.h"
#include "hdtSkyrimPhysicsWorld.h"
#include "Hooks.h"
#include "HookEvents.h"

namespace hdt
{
	static LogToFile s_Log("Data/SKSE/Plugins/hdtSSEPhysics.log");

	bool menuFilter(const char* menuName)
	{
		static const std::string menuFilterList[] =
		{
			"Crafting Menu",
			"Dialogue Menu",
			"RaceSex Menu",
			"HUD Menu",
			"Cursor Menu",
			//"Fader Menu"
		};

		auto iter = std::find(std::begin(menuFilterList), std::end(menuFilterList), menuName);
		return iter != std::end(menuFilterList);
	}

	class FreezeEventHandler : public BSTEventSink <MenuOpenCloseEvent>
	{
	public:
		FreezeEventHandler() {}

		virtual EventResult ReceiveEvent(MenuOpenCloseEvent * evn, EventDispatcher<MenuOpenCloseEvent> * dispatcher) override
		{
			if (menuFilter(evn->menuName))
				return kEvent_Continue;

			std::lock_guard<decltype(m_lock)> l(m_lock);

			if (evn->opening)
			{
				m_menuList.push_back(evn->menuName.data);
				LogDebug("Push Menu : %s", evn->menuName.data);

				if (m_menuList.size() == 1)
				{
					LogDebug("Suspend World");
					SkyrimPhysicsWorld::get()->suspend();
				}
			}
			else
			{
				auto idx = std::find(m_menuList.begin(), m_menuList.end(), evn->menuName.data);
				if (idx != m_menuList.end())
				{
					m_menuList.erase(idx);
					LogDebug("Pop Menu : %s", evn->menuName.data);

					if (!strcmp(evn->menuName.data, "LoadWaitSpinner"))
						SkyrimPhysicsWorld::get()->resetSystems();

					if (m_menuList.empty())
					{
						LogDebug("Resume World");
						SkyrimPhysicsWorld::get()->resume();
					}
				}
			}

			return kEvent_Continue;
		}

		std::mutex m_lock;
		std::vector<std::string> m_menuList;
	} g_freezeEventHandler;

	bool SMPDebug_Execute(const ObScriptParam* paramInfo, ScriptData* scriptData, TESObjectREFR* thisObj, TESObjectREFR* containingObj, Script* scriptObj, ScriptLocals* locals, double& result, UInt32& opcodeOffsetPtr)
	{
		auto skeletons = ArmorManager::instance()->getSkeletons();

		size_t activeSkeletons = 0;
		size_t armors = 0;
		size_t activeArmors = 0;
		
		for (auto skeleton : skeletons)
		{
			if (skeleton.isActiveInScene())
				activeSkeletons++;

			for (const auto armor : skeleton.armors)
			{
				armors++;

				if (armor.physics && armor.physics->m_world)
					activeArmors++;
			}
		}
		
		Console_Print("[HDT-SMP] tracked skeletons: %d", skeletons.size());
		Console_Print("[HDT-SMP] active skeletons: %d", activeSkeletons);
		Console_Print("[HDT-SMP] tracked armors: %d", armors);
		Console_Print("[HDT-SMP] active armors: %d", activeArmors);
		
		return true;
	}
}

extern "C"
{
	bool SKSEPlugin_Query(const SKSEInterface * skse, PluginInfo * info)
	{
		// populate info structure
		info->infoVersion = PluginInfo::kInfoVersion;
		info->name = "hdtSSEPhysics";
		info->version = 1;

		if (skse->isEditor)
		{
			return false;
		}

		if (skse->runtimeVersion != CURRENT_RELEASE_RUNTIME)
			return false;

		if (!hdt::getFramework())
		{
			hdt::LogError("hdtSSEFramework failed to launch");
			return false;
		}

		if (!hdt::getFramework()->isSupportedSkyrimVersion(skse->runtimeVersion))
		{
			hdt::LogError("hdtSSEFramework doesn't support current skyrim version");
			return false;
		}

		if (!hdt::checkFrameworkVersion(1, 1))
		{
			return false;
		}
		return true;
	}

	bool SKSEPlugin_Load(const SKSEInterface * skse)
	{
		hdt::g_frameEventDispatcher.addListener(hdt::ArmorManager::instance());
		hdt::g_frameEventDispatcher.addListener(hdt::SkyrimPhysicsWorld::get());
		hdt::g_shutdownEventDispatcher.addListener(hdt::ArmorManager::instance());
		hdt::g_shutdownEventDispatcher.addListener(hdt::SkyrimPhysicsWorld::get());
		hdt::g_armorAttachEventDispatcher.addListener(hdt::ArmorManager::instance());

		hdt::hookAll();

		const auto messageInterface = reinterpret_cast<SKSEMessagingInterface*>(skse->QueryInterface(kInterface_Messaging));
		if (messageInterface)
		{
			messageInterface->RegisterListener(skse->GetPluginHandle(), "SKSE", [](SKSEMessagingInterface::Message* msg)
			{
				if (msg && msg->type == SKSEMessagingInterface::kMessage_InputLoaded)
				{
					MenuManager * mm = MenuManager::GetSingleton();
					if (mm)
						mm->MenuOpenCloseEventDispatcher()->AddEventSink(&hdt::g_freezeEventHandler);
					hdt::loadConfig();
				}
			});
		}

		ObScriptCommand* hijackedCommand = nullptr;
		for (ObScriptCommand* iter = g_firstConsoleCommand; iter->opcode < kObScript_NumConsoleCommands + kObScript_ConsoleOpBase; ++iter)
		{
			if (!strcmp(iter->longName, "ShowRenderPasses"))
			{
				hijackedCommand = iter;
				break;
			}
		}
		if (hijackedCommand)
		{
			ObScriptCommand cmd = *hijackedCommand;
			cmd.longName = "SMPDebug";
			cmd.shortName = "smp";
			cmd.helpText = "smp";
			cmd.needsParent = 0;
			cmd.numParams = 0;
			cmd.execute = hdt::SMPDebug_Execute;
			cmd.flags = 0;
			SafeWriteBuf(reinterpret_cast<uintptr_t>(hijackedCommand), &cmd, sizeof(cmd));
		}

		return true;
	}
}
