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
			"Cursor Menu",
			"LootMenu" // quick loot RE
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

	NiTexturePtr* GetTextureFromIndex(BSLightingShaderMaterial* material, UInt32 index)
	{
		switch (index)
		{
		case 0:
			return &material->texture1;
			break;
		case 1:
			return &material->texture2;
			break;
		case 2:
		{
			if (material->GetShaderType() == BSShaderMaterial::kShaderType_FaceGen)
			{
				return &static_cast<BSLightingShaderMaterialFacegen*>(material)->unkB0;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_GlowMap)
			{
				return &static_cast<BSLightingShaderMaterialFacegen*>(material)->unkB0;
			}
			else
			{
				return &material->texture3;
			}
		}
		break;
		case 3:
		{
			if (material->GetShaderType() == BSShaderMaterial::kShaderType_FaceGen)
			{
				return &static_cast<BSLightingShaderMaterialFacegen*>(material)->unkA8;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_Parallax)
			{
				return &static_cast<BSLightingShaderMaterialParallax*>(material)->unkA0;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_Parallax || material->GetShaderType() == BSShaderMaterial::kShaderType_ParallaxOcc)
			{
				return &static_cast<BSLightingShaderMaterialParallaxOcc*>(material)->unkA0;
			}
		}
		break;
		case 4:
		{
			if (material->GetShaderType() == BSShaderMaterial::kShaderType_Eye)
			{
				return &static_cast<BSLightingShaderMaterialEye*>(material)->unkA0;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_EnvironmentMap)
			{
				return &static_cast<BSLightingShaderMaterialEnvmap*>(material)->unkA0;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_MultilayerParallax)
			{
				return &static_cast<BSLightingShaderMaterialMultiLayerParallax*>(material)->unkA8;
			}
		}
		break;
		case 5:
		{
			if (material->GetShaderType() == BSShaderMaterial::kShaderType_Eye)
			{
				return &static_cast<BSLightingShaderMaterialEye*>(material)->unkA8;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_EnvironmentMap)
			{
				return &static_cast<BSLightingShaderMaterialEnvmap*>(material)->unkA0;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_MultilayerParallax)
			{
				return &static_cast<BSLightingShaderMaterialMultiLayerParallax*>(material)->unkB0;
			}
		}
		break;
		case 6:
		{
			if (material->GetShaderType() == BSShaderMaterial::kShaderType_FaceGen)
			{
				return &static_cast<BSLightingShaderMaterialFacegen*>(material)->renderedTexture;
			}
			else if (material->GetShaderType() == BSShaderMaterial::kShaderType_MultilayerParallax)
			{
				return &static_cast<BSLightingShaderMaterialMultiLayerParallax*>(material)->unkA0;
			}
		}
		break;
		case 7:
			return &material->texture4;
			break;
		}

		return nullptr;
	}
	
	void DumpNodeChildren(NiAVObject* node)
	{
		_MESSAGE("{%s} {%s} {%X}", node->GetRTTI()->name, node->m_name, node);
		if (node->m_extraDataLen > 0) {
			gLog.Indent();
			for (UInt16 i = 0; i < node->m_extraDataLen; i++) {
				_MESSAGE("{%s} {%s} {%X}", node->m_extraData[i]->GetRTTI()->name, node->m_extraData[i]->m_pcName, node);
			}
			gLog.Outdent();
		}

		NiNode* niNode = node->GetAsNiNode();
		if (niNode && niNode->m_children.m_emptyRunStart > 0)
		{
			gLog.Indent();
			for (int i = 0; i < niNode->m_children.m_emptyRunStart; i++)
			{
				NiAVObject* object = niNode->m_children.m_data[i];
				if (object) {
					NiNode* childNode = object->GetAsNiNode();
					BSGeometry* geometry = object->GetAsBSGeometry();
					if (geometry) {
						_MESSAGE("{%s} {%s} {%X} - Geometry", object->GetRTTI()->name, object->m_name, object);
						NiPointer<BSShaderProperty> shaderProperty = niptr_cast<BSShaderProperty>(geometry->m_spEffectState);
						if (shaderProperty) {
							BSLightingShaderProperty* lightingShader = ni_cast(shaderProperty, BSLightingShaderProperty);
							if (lightingShader) {
								BSLightingShaderMaterial* material = (BSLightingShaderMaterial*)lightingShader->material;

								gLog.Indent();
								for (int i = 0; i < BSTextureSet::kNumTextures; ++i)
								{
									const char* texturePath = material->textureSet->GetTexturePath(i);
									if (!texturePath) {
										continue;
									}

									const char* textureName = "";
									NiTexturePtr* texture = GetTextureFromIndex(material, i);
									if (texture) {
										textureName = texture->get()->name;
									}

									_MESSAGE("Texture %d - %s (%s)", i, texturePath, textureName);
								}

								gLog.Outdent();
							}
						}
					}
					else if (childNode) {
						DumpNodeChildren(childNode);
					}
					else {
						_MESSAGE("{%s} {%s} {%X}", object->GetRTTI()->name, object->m_name, object);
					}
				}
			}
			gLog.Outdent();
		}
	}

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

#ifdef DEBUG
		if (thisObj)
			DumpNodeChildren(thisObj->GetNiRootNode(0));
#endif

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
#ifdef DEBUG
						hdt::g_armorAttachEventDispatcher.addListener(&hdt::g_eventDebugLogger);
						GetEventDispatcherList()->unk1B8.AddEventSink(&hdt::g_eventDebugLogger);
						GetEventDispatcherList()->unk840.AddEventSink(&hdt::g_eventDebugLogger);
#endif
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
