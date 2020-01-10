#include "skse64/GameReferences.h"

#include "ActorManager.h"
#include "hdtSkyrimPhysicsWorld.h"
#include "hdtDefaultBBP.h"
#include "skse64/GameRTTI.h"
#include "skse64/NiSerialization.h"
#include <cinttypes>
#include "Offsets.h"
#include "skse64/GameStreams.h"

namespace hdt
{
	ActorManager::ActorManager()
	{
	}


	ActorManager::~ActorManager()
	{
	}

	ActorManager* ActorManager::instance()
	{
		static ActorManager s;
		return &s;
	}

	IDStr ActorManager::generatePrefix(NiAVObject* armor)
	{
		char buffer[128];
		sprintf_s(buffer, "hdtSSEPhysics_AutoRename_%016llX ", (uintptr_t)armor);
		return IDStr(buffer);
	}

	inline bool isFirstPersonSkeleton(NiNode* npc)
	{
		if (!npc) return false;
		if (findNode(npc, "Camera1st [Cam1]")) return true;
		return false;
	}

	void ActorManager::onEvent(const ArmorAttachEvent& e)
	{
		auto npc = findNode(e.skeleton, "NPC");

		if (!npc) return;

		// TODO: replace this with a generic skeleton fixing configuration option
		// hardcode an exception for lurker skeletons because they are made incorrectly
		if (e.skeleton->m_owner && e.skeleton->m_owner->baseForm)
		{
			auto npcForm = DYNAMIC_CAST(e.skeleton->m_owner->baseForm, TESForm, TESNPC);
			if (npcForm && npcForm->race.race)
			{
				if (!strcmp(npcForm->race.race->models[0].GetModelName(),
				            "Actors\\DLC02\\BenthicLurker\\Character Assets\\skeleton.nif"))
				{
					npc = findNode(e.skeleton, "NPC Root [Root]");
				}
			}
		}

		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		if (e.hasAttached)
		{
			auto prefix = generatePrefix(e.armorModel);
			auto& skeleton = getSkeletonData(e.skeleton);
			auto iter = std::find_if(skeleton.armors.begin(), skeleton.armors.end(), [=](Armor& i)
			{
				return i.prefix == prefix;
			});

			if (iter != skeleton.armors.end())
			{
				iter->armorWorn = e.attachedNode;
				std::unordered_map<IDStr, IDStr> renameMap = iter->renameMap;

				if (!isFirstPersonSkeleton(e.skeleton))
				{
					auto system = SkyrimSystemCreator().createSystem(npc, e.attachedNode, iter->physicsFile,
					                                          std::move(renameMap));

					if (system)
					{
						SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(system);
						iter->physics = system;
					}
				}
			}
		}
		else
		{
			auto prefix = generatePrefix(e.armorModel);
			auto& skeleton = getSkeletonData(e.skeleton);
			skeleton.npc = npc;
			auto iter = std::find_if(skeleton.armors.begin(), skeleton.armors.end(), [=](Armor& i)
			{
				return i.prefix == prefix;
			});
			if (iter == skeleton.armors.end())
			{
				skeleton.armors.push_back(Armor());
				skeleton.armors.back().prefix = prefix;
				iter = skeleton.armors.end() - 1;
			}
			Skeleton::doSkeletonMerge(npc, e.armorModel, prefix, iter->renameMap);
			iter->physicsFile = scanBBP(e.armorModel);
		}
	}

	void ActorManager::reloadMeshes()
	{
		const FrameEvent e
		{
			false
		};

		onEvent(e);

		for (auto& i : m_skeletons)
		{
			for (auto& j : i.armors)
			{
				if (j.physics && j.physics->m_world)
					j.physics->m_world->removeSkinnedMeshSystem(j.physics);

				j.physics = nullptr;

				std::unordered_map<IDStr, IDStr> renameMap = j.renameMap;

				auto system = SkyrimSystemCreator().createSystem(i.npc, j.armorWorn, j.physicsFile, std::move(renameMap));

				if (system)
				{
					SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(system);
					j.physics = system;
				}
			}

			for (auto& headPart : i.head.headParts)
			{
				if (headPart.physics && headPart.physics->m_world)
					headPart.physics->m_world->removeSkinnedMeshSystem(headPart.physics);

				headPart.physics = nullptr;

				std::unordered_map<IDStr, IDStr> renameMap = i.head.renameMap;

				auto system = SkyrimSystemCreator().createSystem(i.npc, i.head.headNode, headPart.physicsFile,
				                                          std::move(renameMap));

				if (system)
				{
					SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(system);
					headPart.physics = system;
				}
			}
		}
	}

	void ActorManager::onEvent(const FrameEvent& e)
	{
		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		for (auto& i : m_skeletons)
		{
			if (!i.isActiveInScene())
			{
				if (i.skeleton->m_uiRefCount == 1)
				{
					i.clear();
					i.skeleton = nullptr;
				}
				else
				{
					for (auto& j : i.armors)
					{
						if (j.physics && j.physics->m_world)
							j.physics->m_world->removeSkinnedMeshSystem(j.physics);
					}

					for (auto& j : i.head.headParts)
					{
						if (j.physics && j.physics->m_world)
							j.physics->m_world->removeSkinnedMeshSystem(j.physics);
					}
				}
			}
			else
			{
				auto world = SkyrimPhysicsWorld::get();
				for (auto& j : i.armors)
				{
					if (j.physics && !j.physics->m_world)
						world->addSkinnedMeshSystem(j.physics);
				}

				for (auto& j : i.head.headParts)
				{
					if (j.physics && !j.physics->m_world)
						world->addSkinnedMeshSystem(j.physics);
				}
			}
		}
		m_skeletons.erase(
			std::remove_if(m_skeletons.begin(), m_skeletons.end(), [](Skeleton& i) { return !i.skeleton; }),
			m_skeletons.end());

		for (auto& i : m_skeletons)
		{
			i.cleanArmor();
			i.cleanHead();
		}
	}

	void ActorManager::onEvent(const ShutdownEvent&)
	{
		m_shutdown = true;
		std::lock_guard<decltype(m_lock)> l(m_lock);

		m_skeletons.clear();
	}

	void ActorManager::onEvent(const SkinSingleHeadGeometryEvent& e)
	{
		auto npc = findNode(e.skeleton, "NPC");

		if (!npc) return;

		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		auto& skeleton = getSkeletonData(e.skeleton);
		skeleton.npc = npc;

		if (e.hasSkinned)
		{
			auto headPartIter = std::find_if(skeleton.head.headParts.begin(), skeleton.head.headParts.end(),
			                                 [e](const Head::HeadPart& p)
			                                 {
				                                 return p.headPart == e.geometry;
			                                 });

			if (headPartIter != skeleton.head.headParts.end())
			{
				if (headPartIter->origPartRootNode)
				{
					_DMESSAGE("renaming nodes in original part %s back", headPartIter->origPartRootNode->m_name);
					for (auto& entry : skeleton.head.renameMap)
					{
						auto node = findNode(headPartIter->origPartRootNode, entry.second->cstr());
						if (node)
						{
							_DMESSAGE("rename node %s -> %s", entry.second->cstr(), entry.first->cstr());
							setNiNodeName(node, entry.first->cstr());
						}
					}
				}
				headPartIter->origPartRootNode = nullptr;
			}

			if (!skeleton.head.isFullSkinning)
				skeleton.scanHead();
		}
		else if (!e.hasSkinned)
		{
			skeleton.updateHead(e.headNode, e.geometry);
		}
	}

	void ActorManager::onEvent(const SkinAllHeadGeometryEvent& e)
	{
		auto npc = findNode(e.skeleton, "NPC");

		if (!npc) return;

		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		auto& skeleton = getSkeletonData(e.skeleton);
		skeleton.npc = npc;
		if (e.skeleton->m_owner)
			skeleton.skeletonOwner = e.skeleton->m_owner;

		if (e.hasSkinned)
		{
			skeleton.scanHead();
			skeleton.head.isFullSkinning = false;
			if (skeleton.head.npcFaceGeomNode)
			{
				_DMESSAGE("npc face geom no longer needed, clearing ref");
				skeleton.head.npcFaceGeomNode = nullptr;
			}
		}
		else
		{
			skeleton.head.isFullSkinning = true;
		}
	}

	std::vector<ActorManager::Skeleton> ActorManager::getSkeletons() const
	{
		return m_skeletons;
	}

	ActorManager::Skeleton& ActorManager::getSkeletonData(NiNode* skeleton)
	{
		auto iter = std::find_if(m_skeletons.begin(), m_skeletons.end(), [=](Skeleton& i)
		{
			return i.skeleton == skeleton;
		});
		if (iter != m_skeletons.end())
		{
			return *iter;
		}
		if (!isFirstPersonSkeleton(skeleton))
		{
			auto ownerIter = std::find_if(m_skeletons.begin(), m_skeletons.end(), [=](Skeleton& i)
			{
				return !isFirstPersonSkeleton(i.skeleton) && i.skeletonOwner && skeleton->m_owner && i.skeletonOwner ==
					skeleton->m_owner;
			});
			if (ownerIter != m_skeletons.end())
			{
				_DMESSAGE("new skeleton found for formid %08x", skeleton->m_owner->formID);
				ownerIter->cleanHead(true);
			}
		}
		m_skeletons.push_back(Skeleton());
		m_skeletons.back().skeleton = skeleton;
		return m_skeletons.back();
	}

	void ActorManager::Skeleton::doSkeletonMerge(NiNode* dst, NiNode* src, IString* prefix,
	                                             std::unordered_map<IDStr, IDStr>& map)
	{
		for (int i = 0; i < src->m_children.m_arrayBufLen; ++i)
		{
			auto srcChild = castNiNode(src->m_children.m_data[i]);
			if (!srcChild) continue;

			if (!srcChild->m_name)
			{
				doSkeletonMerge(dst, srcChild, prefix, map);
				continue;
			}

			auto dstChild = findNode(dst, srcChild->m_name);
			if (dstChild)
			{
				doSkeletonMerge(dstChild, srcChild, prefix, map);
			}
			else
			{
				dst->AttachChild(cloneNodeTree(srcChild, prefix, map), false);
			}
		}
	}

	NiNode* ActorManager::Skeleton::cloneNodeTree(NiNode* src, IString* prefix, std::unordered_map<IDStr, IDStr>& map)
	{
		NiCloningProcess c;
		auto ret = static_cast<NiNode*>(src->CreateClone(c));
		src->ProcessClone(&c);

		renameTree(src, prefix, map);
		renameTree(ret, prefix, map);

		return ret;
	}

	void ActorManager::Skeleton::renameTree(NiNode* root, IString* prefix, std::unordered_map<IDStr, IDStr>& map)
	{
		if (root->m_name)
		{
			std::string newName(prefix->cstr(), prefix->size());
			newName += root->m_name;
			if (map.insert(std::make_pair<IDStr, IDStr>(root->m_name, newName)).second)
				_DMESSAGE("Rename Bone %s -> %s", root->m_name, newName.c_str());
			setNiNodeName(root, newName.c_str());
		}

		for (int i = 0; i < root->m_children.m_arrayBufLen; ++i)
		{
			auto child = castNiNode(root->m_children.m_data[i]);
			if (child)
				renameTree(child, prefix, map);
		}
	}

	void ActorManager::Skeleton::doSkeletonClean(NiNode* dst, IString* prefix)
	{
		for (int i = dst->m_children.m_arrayBufLen - 1; i >= 0; --i)
		{
			auto child = castNiNode(dst->m_children.m_data[i]);
			if (!child) continue;

			if (child->m_name && !strncmp(child->m_name, prefix->cstr(), prefix->size()))
			{
				dst->RemoveAt(i++);
			}
			else
			{
				doSkeletonClean(child, prefix);
			}
		}
	}

	void ActorManager::Skeleton::cleanArmor()
	{
		for (auto& i : armors)
		{
			if (!i.armorWorn) continue;
			if (i.armorWorn->m_parent) continue;

			SkyrimPhysicsWorld::get()->removeSkinnedMeshSystem(i.physics);
			if (npc) doSkeletonClean(npc, i.prefix);
			i.prefix = nullptr;
		}

		armors.erase(std::remove_if(armors.begin(), armors.end(), [](Armor& i) { return !i.prefix; }), armors.end());
	}

	void ActorManager::Skeleton::cleanHead(bool cleanAll)
	{
		for (auto& headPart : head.headParts)
		{
			if (!headPart.headPart->m_parent || cleanAll)
			{
				if (!cleanAll)
					_DMESSAGE("headpart %s disconnected", headPart.headPart->m_name);
				else
					_DMESSAGE("cleaning headpart %s due to clean all", headPart.headPart->m_name);

				auto renameIt = this->head.renameMap.begin();

				while (renameIt != this->head.renameMap.end())
				{
					bool erase = false;

					_DMESSAGE("count %d", headPart.renamedBonesInUse.size());
					if (headPart.renamedBonesInUse.count(renameIt->first) != 0)
					{
						auto findNode = this->head.nodeUseCount.find(renameIt->first);
						if (findNode != this->head.nodeUseCount.end())
						{
							findNode->second -= 1;
							_DMESSAGE("decrementing use count by 1, it is now %d", findNode->second);
							if (findNode->second <= 0)
							{
								_DMESSAGE("node no longer in use, cleaning from skeleton");
								auto removeObj = findObject(npc, renameIt->second->cstr());
								if (removeObj)
								{
									_DMESSAGE("found node %s, removing", removeObj->m_name);
									auto parent = removeObj->m_parent;
									if (parent)
									{
										parent->RemoveChild(removeObj);
										removeObj->DecRef();
									}
								}
								this->head.nodeUseCount.erase(findNode);
								erase = true;
							}
						}
					}

					if (erase)
						renameIt = this->head.renameMap.erase(renameIt);
					else
						++renameIt;
				}

				headPart.headPart = nullptr;
				headPart.origPartRootNode = nullptr;
				if (headPart.physics)
					SkyrimPhysicsWorld::get()->removeSkinnedMeshSystem(headPart.physics);
				headPart.physics = nullptr;
				headPart.renamedBonesInUse.clear();
			}
		}

		head.headParts.erase(std::remove_if(head.headParts.begin(), head.headParts.end(),
		                                    [](Head::HeadPart& i) { return !i.headPart; }), head.headParts.end());
	}

	void ActorManager::Skeleton::clear()
	{
		SkyrimPhysicsWorld::get()->removeSystemByNode(npc);
		cleanHead();
		head.headParts.clear();
		head.headNode = nullptr;
		armors.clear();
	}

	bool ActorManager::Skeleton::isActiveInScene() const
	{
		// TODO: do this better
		// when entering/exiting an interior NPCs are detached from the scene but not unloaded, so we need to check two levels up 
		// this properly removes exterior cell armors from the physics world when entering an interior, and vice versa
		return skeleton->m_parent && skeleton->m_parent->m_parent && skeleton->m_parent->m_parent->m_parent;
	}


	void ActorManager::Skeleton::doHeadSkeletonMerge(NiNode* dst, NiNode* src, IString* prefix,
	                                                 std::unordered_map<IDStr, IDStr>& map,
	                                                 std::unordered_map<IDStr, uint8_t>& countMap,
													std::set<IDStr>& renamedBonesInUse)
	{
		for (int i = 0; i < src->m_children.m_arrayBufLen; ++i)
		{
			auto srcChild = castNiNode(src->m_children.m_data[i]);
			if (!srcChild) continue;

			if (!srcChild->m_name)
			{
				doHeadSkeletonMerge(dst, srcChild, prefix, map, countMap, renamedBonesInUse);
				continue;
			}

			if (!strcmp(srcChild->m_name, "BSFaceGenNiNodeSkinned"))
			{
				_DMESSAGE("skipping facegen ninode in skeleton merge");
				continue;
			}

			auto dstChild = findNode(dst, srcChild->m_name);
			if (dstChild)
			{
				doHeadSkeletonMerge(dstChild, srcChild, prefix, map, countMap, renamedBonesInUse);
			}
			else
			{
				dst->AttachChild(cloneHeadNodeTree(srcChild, prefix, map, countMap, renamedBonesInUse), false);
			}
		}
	}

	NiNode* ActorManager::Skeleton::cloneHeadNodeTree(NiNode* src, IString* prefix,
	                                                  std::unordered_map<IDStr, IDStr>& map,
	                                                  std::unordered_map<IDStr, uint8_t>& countMap, std::set<IDStr>& renamedBonesInUse)
	{
		NiCloningProcess c;
		auto ret = static_cast<NiNode*>(src->CreateClone(c));
		src->ProcessClone(&c);

		renameHeadTree(src, prefix, map, countMap, renamedBonesInUse);
		renameHeadTree(ret, prefix, map, countMap, renamedBonesInUse);

		return ret;
	}

	void ActorManager::Skeleton::renameHeadTree(NiNode* root, IString* prefix, std::unordered_map<IDStr, IDStr>& map,
	                                            std::unordered_map<IDStr, uint8_t>& countMap, std::set<IDStr>& renamedBonesInUse)
	{
		if (root->m_name)
		{
			std::string newName(prefix->cstr(), prefix->size());
			newName += root->m_name;
			if (map.insert(std::make_pair<IDStr, IDStr>(root->m_name, newName)).second)
			{
				_DMESSAGE("Rename Bone %s -> %s", root->m_name, newName.c_str());
				countMap.insert(std::make_pair<IDStr, uint8_t>(root->m_name, 1));
				renamedBonesInUse.insert(root->m_name);
			}
			setNiNodeName(root, newName.c_str());
		}

		for (int i = 0; i < root->m_children.m_arrayBufLen; ++i)
		{
			auto child = castNiNode(root->m_children.m_data[i]);
			if (child)
				renameHeadTree(child, prefix, map, countMap, renamedBonesInUse);
		}
	}

	void ActorManager::Skeleton::scanHead()
	{
		if (!this->head.headNode)
		{
			_DMESSAGE("actor has no head node");
			return;
		}

		std::unordered_set<std::string> physicsDupes;

		for (auto& headPart : this->head.headParts)
		{
			// always regen physics for all head parts
			if (headPart.physics)
				SkyrimPhysicsWorld::get()->removeSkinnedMeshSystem(headPart.physics);
			headPart.physics = nullptr;

			if (headPart.physicsFile.empty())
			{
				_DMESSAGE("no physics file for headpart %s", headPart.headPart->m_name);
				continue;
			}

			if (physicsDupes.count(headPart.physicsFile))
			{
				_DMESSAGE("previous head part generated physics system for file %s, skipping",
				          headPart.physicsFile.c_str());
				continue;
			}

			std::unordered_map<IDStr, IDStr> renameMap = this->head.renameMap;

			_DMESSAGE("try create system for headpart %s physics file %s", headPart.headPart->m_name,
			          headPart.physicsFile.c_str());
			physicsDupes.insert(headPart.physicsFile);
			auto system = SkyrimSystemCreator().createSystem(npc, this->head.headNode, headPart.physicsFile,
			                                            std::move(renameMap));

			if (system)
			{
				_DMESSAGE("success");
				SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(system);
				headPart.physics = system;
			}
		}
	}

	typedef bool (*_TESNPC_GetFaceGeomPath)(TESNPC* a_npc, char* a_buf);
	RelocAddr<_TESNPC_GetFaceGeomPath> TESNPC_GetFaceGeomPath(offset::TESNPC_GetFaceGeomPath);

	void ActorManager::Skeleton::updateHead(BSFaceGenNiNode* headNode, BSGeometry* geometry)
	{
		if (this->head.headNode && this->head.headNode != headNode)
		{
			_DMESSAGE("completely new head attached to skeleton, clearing tracking");
			for (auto& headPart : this->head.headParts)
			{
				if (headPart.physics)
					SkyrimPhysicsWorld::get()->removeSkinnedMeshSystem(headPart.physics);
				headPart.physics = nullptr;
				headPart.headPart = nullptr;
				headPart.origPartRootNode = nullptr;
			}

			this->head.headParts.clear();

			if (npc)
				doSkeletonClean(npc, this->head.prefix);

			this->head.prefix = nullptr;
			this->head.headNode = nullptr;
			this->head.renameMap.clear();
			this->head.nodeUseCount.clear();
		}

		// clean swapped out headparts
		cleanHead();

		this->head.headNode = headNode;
		this->head.prefix = generatePrefix(headNode);

		auto it = std::find_if(this->head.headParts.begin(), this->head.headParts.end(),
		                       [geometry](const Head::HeadPart& p)
		                       {
			                       return p.headPart == geometry;
		                       });

		if (it != this->head.headParts.end())
		{
			_DMESSAGE("geometry is already added as head part");
			return;
		}

		this->head.headParts.push_back(Head::HeadPart());

		head.headParts.back().headPart = geometry;
		head.headParts.back().physics = nullptr;

		if (skeleton->m_owner && skeleton->m_owner->formID == 0x14) // player character
		{
			auto fmd = static_cast<BSFaceGenModelExtraData*>(geometry->GetExtraData("FMD"));
			if (fmd && fmd->m_model && fmd->m_model->unk10 && fmd->m_model->unk10->unk08)
			{
				const auto facePartRootNode = fmd->m_model->unk10->unk08->GetAsNiNode();
				if (facePartRootNode)
				{
					_DMESSAGE("geometry %s found root node %s", geometry->m_name, facePartRootNode->m_name);
					head.headParts.back().origPartRootNode = facePartRootNode;
					for (auto& entry : this->head.renameMap)
					{
						auto obj = findObject(facePartRootNode, entry.first->cstr());
						if (obj)
						{
							auto node = obj->GetAsNiNode();
							if (node)
							{
								_DMESSAGE("node %s found rename to %s", node->m_name, entry.second->cstr());
								setNiNodeName(node, entry.second->cstr());
								auto findNode = this->head.nodeUseCount.find(entry.first);
								if (findNode != this->head.nodeUseCount.end())
								{
									findNode->second += 1;
									_DMESSAGE("incrementing use count by 1, it is now %d", findNode->second);
									head.headParts.back().renamedBonesInUse.insert(entry.first);
								}
							}
						}
					}
					doHeadSkeletonMerge(npc, facePartRootNode, this->head.prefix, this->head.renameMap,
					                    this->head.nodeUseCount, head.headParts.back().renamedBonesInUse);
					head.headParts.back().physicsFile = scanBBP(facePartRootNode);
				}
			}
		}
		else // npc
		{
			// for NPCs we dont have easy access back to the facegeom .nif so we're loading our own copy, but only do this once
			if (!head.npcFaceGeomNode)
			{
				if (skeleton->m_owner && skeleton->m_owner->baseForm)
				{
					auto npc = DYNAMIC_CAST(skeleton->m_owner->baseForm, TESForm, TESNPC);
					if (npc)
					{
						char filePath[MAX_PATH];
						if (TESNPC_GetFaceGeomPath(npc, filePath))
						{
							_DMESSAGE("loading facegeom from path %s", filePath);
							static const int MAX_SIZE = sizeof(NiStream) + 0x200;
							UInt8 niStreamMemory[MAX_SIZE];
							memset(niStreamMemory, 0, MAX_SIZE);
							NiStream* niStream = (NiStream*)niStreamMemory;
							CALL_MEMBER_FN(niStream, ctor)();
		
							BSResourceNiBinaryStream binaryStream(filePath);
							if (!binaryStream.IsValid())
							{
								_ERROR("somehow npc facegeom was not found");
								CALL_MEMBER_FN(niStream, dtor)();
							}
							else
							{
								niStream->LoadStream(&binaryStream);
								if (niStream->m_rootObjects.m_data[0])
								{
									auto rootFadeNode = niStream->m_rootObjects.m_data[0]->GetAsBSFadeNode();
									if (rootFadeNode)
									{
										_DMESSAGE("npc root fadenode found");
										head.npcFaceGeomNode = rootFadeNode;
										doHeadSkeletonMerge(this->npc, rootFadeNode, this->head.prefix,
										                    this->head.renameMap,
										                    this->head.nodeUseCount, head.headParts.back().renamedBonesInUse);
										head.headParts.back().physicsFile = scanBBP(rootFadeNode);
									}
									else
									{
										_DMESSAGE("npc facegeom root wasn't fadenode as expected");
									}
								}
								CALL_MEMBER_FN(niStream, dtor)();
							}
						}
					}
				}
				else
				{
					_WARNING("skipped npc %08x face part", skeleton->m_owner ? skeleton->m_owner->formID : 0x0);
				}
			}
			else
			{
				_DMESSAGE("npc head already read on previous head part");
			}
		
			if (head.npcFaceGeomNode)
			{
				_DMESSAGE("skinning renamed bones");
				auto obj = findObject(head.npcFaceGeomNode, geometry->m_name);
				if (obj)
				{
					const auto faceGeomGeometry = obj->GetAsBSGeometry();

					if (faceGeomGeometry)
					{
						_DMESSAGE("geometry found in loaded facegeom");
						if (geometry->m_spSkinInstance && geometry->m_spSkinInstance->m_spSkinData)
						{
							for (int i = 0; i < geometry->m_spSkinInstance->m_spSkinData->m_uiBones; i++)
							{
								auto renameIt = head.renameMap.find(faceGeomGeometry->m_spSkinInstance->m_ppkBones[i]->m_name);
								if (renameIt != head.renameMap.end())
								{
									auto node = findNode(npc, renameIt->second->cstr());
									if (node)
									{
										_DMESSAGE("replacing node %s with %s",
											geometry->m_spSkinInstance->m_ppkBones[i]->m_name, node->m_name);
										geometry->m_spSkinInstance->m_ppkBones[i] = node;
										geometry->m_spSkinInstance->m_worldTransforms[i] = &(node->m_worldTransform);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
