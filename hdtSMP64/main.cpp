#include "skse64/GameMenus.h"
#include "skse64/GameReferences.h"
#include "skse64/ObScript.h"
#include "skse64/PluginAPI.h"
#include "skse64_common/skse_version.h"
#include "skse64_common/SafeWrite.h"

#include "ArmorManager.h"
#include "config.h"
#include "EventDebugLogger.h"
#include "hdtSkyrimPhysicsWorld.h"
#include "Hooks.h"
#include "HookEvents.h"

#include <shlobj_core.h>

namespace hdt
{
	IDebugLog	gLog;
	EventDebugLogger g_eventDebugLogger;

	bool menuFilter(const char* menuName)
	{
		static const std::string menuFilterList[] =
		{
			"Crafting Menu",
			"Dialogue Menu",
			"RaceSex Menu",
			"HUD Menu",
			"Cursor Menu"
			//"Fader Menu"
		};

		auto iter = std::find(std::begin(menuFilterList), std::end(menuFilterList), menuName);
		return iter != std::end(menuFilterList);
	}

	class FreezeEventHandler : public BSTEventSink <MenuOpenCloseEvent>
	{
	public:
		FreezeEventHandler() {}

		virtual EventResult ReceiveEvent(MenuOpenCloseEvent* evn, EventDispatcher<MenuOpenCloseEvent>* dispatcher) override
		{
			if (menuFilter(evn->menuName))
				return kEvent_Continue;

			std::lock_guard<decltype(m_lock)> l(m_lock);

			if (evn->opening)
			{
				m_menuList.push_back(evn->menuName.data);
				_DMESSAGE("Push Menu : %s", evn->menuName.data);

				if (m_menuList.size() == 1)
				{
					if (!strcmp(evn->menuName.data, "Loading Menu"))
					{
						_DMESSAGE("Suspend World - Loading Screen detected");
						SkyrimPhysicsWorld::get()->suspend(true);
					}
					else
					{
						_DMESSAGE("Suspend World");
						SkyrimPhysicsWorld::get()->suspend();
					}
				}
			}
			else
			{
				auto idx = std::find(m_menuList.begin(), m_menuList.end(), evn->menuName.data);
				if (idx != m_menuList.end())
				{
					m_menuList.erase(idx);
					_DMESSAGE("Pop Menu : %s", evn->menuName.data);

					if (!strcmp(evn->menuName.data, "LoadWaitSpinner"))
					{
						SkyrimPhysicsWorld::get()->resetSystems();
					}

					if (m_menuList.empty())
					{
						_DMESSAGE("Resume World");									
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
	bool SKSEPlugin_Query(const SKSEInterface* skse, PluginInfo* info)
	{
		// populate info structure
		info->infoVersion = PluginInfo::kInfoVersion;
		info->name = "hdtSSEPhysics";
		info->version = 1;

		hdt::gLog.OpenRelative(CSIDL_MYDOCUMENTS, "\\My Games\\Skyrim Special Edition\\SKSE\\hdtSMP64.log");
		hdt::gLog.SetLogLevel(IDebugLog::LogLevel::kLevel_Message);

		_MESSAGE("hdtSMP64 2.0");

		if (skse->isEditor)
		{
			return false;
		}

		if (skse->runtimeVersion != CURRENT_RELEASE_RUNTIME)
		{
			_FATALERROR("attempted to load plugin into unsupported game version, exiting");
			return false;
		}

		return true;
	}

	bool SKSEPlugin_Load(const SKSEInterface* skse)
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
						MenuManager* mm = MenuManager::GetSingleton();
						if (mm)
							mm->MenuOpenCloseEventDispatcher()->AddEventSink(&hdt::g_freezeEventHandler);
						hdt::loadConfig();
						//hdt::g_armorAttachEventDispatcher.addListener(&hdt::g_eventDebugLogger);
						//GetEventDispatcherList()->unk1B8.AddEventSink(&hdt::g_eventDebugLogger);
						//GetEventDispatcherList()->unk840.AddEventSink(&hdt::g_eventDebugLogger);
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
