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

	IDStr ActorManager::armorPrefix(ActorManager::IDType id)
	{
		char buffer[128];
		sprintf_s(buffer, "hdtSSEPhysics_AutoRename_Armor_%08X ", id);
		return IDStr(buffer);
	}

	IDStr ActorManager::headPrefix(ActorManager::IDType id)
	{
		char buffer[128];
		sprintf_s(buffer, "hdtSSEPhysics_AutoRename_Head_%08X ", id);
		return IDStr(buffer);
	}

	inline bool isFirstPersonSkeleton(NiNode* npc)
	{
		if (!npc) return false;
		return findNode(npc, "Camera1st [Cam1]") ? true : false;
	}

	NiNode* getNpcNode(NiNode* skeleton)
	{
		// TODO: replace this with a generic skeleton fixing configuration option
		// hardcode an exception for lurker skeletons because they are made incorrectly
		auto shouldFix = false;
		if (skeleton->m_owner && skeleton->m_owner->baseForm)
		{
			auto npcForm = DYNAMIC_CAST(skeleton->m_owner->baseForm, TESForm, TESNPC);
			if (npcForm && npcForm->race.race
				&& !strcmp(npcForm->race.race->models[0].GetModelName(), "Actors\\DLC02\\BenthicLurker\\Character Assets\\skeleton.nif"))
				shouldFix = true;
		}
		return findNode(skeleton, shouldFix ? "NPC Root [Root]" : "NPC");
	}

	// TODO Shouldn't there be an ArmorDetachEvent?
	void ActorManager::onEvent(const ArmorAttachEvent& e)
	{
		// No armor is ever attached to a lurker skeleton, thus we don't need to test.
		if (e.skeleton == nullptr || !findNode(e.skeleton, "NPC"))
		{
			return;
		}

		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		auto& skeleton = getSkeletonData(e.skeleton);
		if (e.hasAttached)
		{
			// Check override data for current armoraddon
			if (e.skeleton->m_owner)
			{
				auto actor_formID = e.skeleton->m_owner->formID;
				if (actor_formID) {
					auto old_physics_file = skeleton.getArmors().back().physicsFile.first;
					std::string physics_file_path_override = hdt::Override::OverrideManager::GetSingleton()->checkOverride(actor_formID, old_physics_file);
					if (!physics_file_path_override.empty()) {
						//Console_Print("[DynamicHDT] -- ArmorAddon %s is overridden ", e.attachedNode->m_name);
						skeleton.getArmors().back().physicsFile.first = physics_file_path_override;
					}
				}
			}


			skeleton.attachArmor(e.armorModel, e.attachedNode);
		}
		else
		{
			skeleton.addArmor(e.armorModel);
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
			i.reloadMeshes();
		}
	}

	void ActorManager::onEvent(const FrameEvent& e)
	{
		// Other events have to be managed. The FrameEvent is the only event that we can drop,
		// we always have one later where we'll be able to manage the passed time.
		// We drop this execution when the lock is already taken; in that case, we would execute the code later.
		// It is better to drop it now, and let the next frame manage it.
		// Moreover, dropping a locked part of the code allows to reduce the total wait times.
		// Finally, some skse mods issue FrameEvents, this mechanism manages the case where they issue too many.
		std::unique_lock<decltype(m_lock)> lock(m_lock, std::try_to_lock);
		if (!lock.owns_lock()) return;

		if (m_shutdown) return;

		// We get the player character and its cell.
		// TODO Isn't there a more performing way to find the PC?? A singleton? And if it's the right way, why isn't it in utils functions?
		auto& playerCharacter = std::find_if(m_skeletons.begin(), m_skeletons.end(), [](Skeleton& s) { return s.isPlayerCharacter(); });
		auto playerCell = (playerCharacter != m_skeletons.end() && playerCharacter->skeleton->m_parent) ? playerCharacter->skeleton->m_parent->m_parent : nullptr;

		// We get the camera, its position and orientation.
#ifndef SKYRIMVR
		auto camera = PlayerCamera::GetSingleton()->cameraNode->m_worldTransform;
		auto cameraPosition = camera.pos;
		auto cameraOrientation = camera.rot * NiPoint3(0., 1., 0.); // The camera matrix is relative to the world.
#else
		//camera info taken from Shizof's cpbc under MIT. https://www.nexusmods.com/skyrimspecialedition/mods/21224?tab=files
		//commented out translations were to put the camera behind the player to widen fov and aren't needed
		if (!(*g_thePlayer)->loadedState)
			return;
		auto cameraPosition = (*g_thePlayer)->loadedState->node->m_worldTransform.pos;// +((*g_thePlayer)->loadedState->node->m_worldTransform.rot * NiPoint3(0, -200.0f, 0));
		auto cameraOrientation = /*cameraPosition +*/ ((*g_thePlayer)->loadedState->node->m_worldTransform.rot * NiPoint3(0, 1.0f, 0));
#endif

		for (auto& i : m_skeletons)
		{
			if (i.skeleton->m_uiRefCount == 1)
			{
				i.clear();
				i.skeleton = nullptr;
			}
			else if (i.hasPhysics)
				// TODO Ease of configuration: maxAngle could be autocalculated depending on the FOV.
				i.updateAttachedState(cameraPosition, m_maxDistance2, playerCell, cameraOrientation, m_cosMaxAngle2);
		}

		m_skeletons.erase(
			std::remove_if(m_skeletons.begin(), m_skeletons.end(), [](Skeleton& i) { return !i.skeleton; }),
			m_skeletons.end());

		for (auto& i : m_skeletons)
		{
			i.cleanArmor();
			i.cleanHead();
		}
		if (!SkyrimPhysicsWorld::get()->isSuspended() && updateCount++ % SkyrimPhysicsWorld::get()->min_fps == 0) {
			auto processing_time = SkyrimPhysicsWorld::get()->m_averageProcessingTime;
			auto target_time = SkyrimPhysicsWorld::get()->m_timeTick * 250;
			_MESSAGE("active %d maxTrackedSkeletons %d numIterations %d groupIterations %d processTime %f target_time %f m_substepTick %f", activeSkeletons, maxTrackedSkeletons, SkyrimPhysicsWorld::get()->getSolverInfo().m_numIterations, ConstraintGroup::MaxIterations, processing_time, target_time, SkyrimPhysicsWorld::get()->m_substepTick * 1000);
			updateCount = 1;
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
		// This case never happens to a lurker skeleton, thus we don't need to test.
		auto npc = findNode(e.skeleton, "NPC");
		if (!npc) return;

		std::lock_guard<decltype(m_lock)> l(m_lock);
		if (m_shutdown) return;

		auto& skeleton = getSkeletonData(e.skeleton);
		skeleton.npc = getNpcNode(e.skeleton);

		skeleton.processGeometry(e.headNode, e.geometry);

		auto headPartIter = std::find_if(skeleton.head.headParts.begin(), skeleton.head.headParts.end(),
			[e](const Head::HeadPart& p)
			{
				return p.headPart == e.geometry;
			});

		if (headPartIter != skeleton.head.headParts.end())
		{
			if (headPartIter->origPartRootNode)
			{

#ifdef _DEBUG
				_DMESSAGE("renaming nodes in original part %s back", headPartIter->origPartRootNode->m_name);
#endif // _DEBUG

				for (auto& entry : skeleton.head.renameMap)
				{
					// This case never happens to a lurker skeleton, thus we don't need to test.
					auto node = findNode(headPartIter->origPartRootNode, entry.second->cstr());
					if (node)
					{
#ifdef _DEBUG
						_DMESSAGE("rename node %s -> %s", entry.second->cstr(), entry.first->cstr());
#endif // _DEBUG
						setNiNodeName(node, entry.first->cstr());
					}
				}
			}
			headPartIter->origPartRootNode = nullptr;
		}

		if (!skeleton.head.isFullSkinning)
			skeleton.scanHead();
	}

	void ActorManager::onEvent(const SkinAllHeadGeometryEvent& e)
	{
		// This case never happens to a lurker skeleton, thus we don't need to test.
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
#ifdef _DEBUG
				_DMESSAGE("npc face geom no longer needed, clearing ref");
#endif // _DEBUG
				skeleton.head.npcFaceGeomNode = nullptr;
			}
		}
		else
		{
			skeleton.head.isFullSkinning = true;
		}
	}

	void ActorManager::PhysicsItem::setPhysics(Ref<SkyrimSystem>& system, bool active)
	{
		clearPhysics();
		m_physics = system;
		if (active)
		{
			SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(m_physics);
		}
	}

	void ActorManager::PhysicsItem::clearPhysics()
	{
		if (state() == ItemState::e_Active)
		{
			m_physics->m_world->removeSkinnedMeshSystem(m_physics);
		}
		m_physics = nullptr;
	}

	ActorManager::ItemState ActorManager::PhysicsItem::state() const
	{
		return m_physics ? (m_physics->m_world ? ItemState::e_Active : ItemState::e_Inactive) : ItemState::e_NoPhysics;
	}

	const std::vector<Ref<SkinnedMeshBody>>& ActorManager::PhysicsItem::meshes() const
	{
		return m_physics->meshes();
	}

	void ActorManager::PhysicsItem::updateActive(bool active)
	{
		if (active && state() == ItemState::e_Inactive)
		{
			SkyrimPhysicsWorld::get()->addSkinnedMeshSystem(m_physics);
		}
		else if (!active && state() == ItemState::e_Active)
		{
			m_physics->m_world->removeSkinnedMeshSystem(m_physics);
		}
	}

	std::vector<ActorManager::Skeleton>& ActorManager::getSkeletons()
	{
		return m_skeletons;
	}

#ifdef ANNIVERSARY_EDITION
	bool ActorManager::skeletonNeedsParts(NiNode* skeleton)
	{
		return !isFirstPersonSkeleton(skeleton);
		/*
		auto iter = std::find_if(m_skeletons.begin(), m_skeletons.end(), [=](Skeleton& i)
		{
			return i.skeleton == skeleton;
		});
		if (iter != m_skeletons.end())
		{
			return (iter->head.headNode == 0);
		}
		*/
	}
#endif
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
#ifdef _DEBUG
				_DMESSAGE("new skeleton found for formid %08x", skeleton->m_owner->formID);
#endif // _DEBUG
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

			// FIXME: This was previously only in doHeadSkeletonMerge.
			// But surely non-head skeletons wouldn't have this anyway?
			if (!strcmp(srcChild->m_name, "BSFaceGenNiNodeSkinned"))
			{
#ifdef _DEBUG
				_DMESSAGE("skipping facegen ninode in skeleton merge");
#endif // _DEBUG
				continue;
			}

			// TODO check it's not a lurker skeleton
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

		// FIXME: cloneHeadNodeTree just did this for ret, not both. Don't know if that matters. Armor parts need it on both.
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
#ifdef _DEBUG
			if (map.insert(std::make_pair<IDStr, IDStr>(root->m_name, newName)).second)
				_DMESSAGE("Rename Bone %s -> %s", root->m_name, newName.c_str());
#else
			map.insert(std::make_pair<IDStr, IDStr>(root->m_name, newName));
#endif // _DEBUG

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

	// returns the name of the skeleton owner
	std::string ActorManager::Skeleton::name()
	{
		if (skeleton->m_owner && skeleton->m_owner->baseForm) {
			auto bname = DYNAMIC_CAST(skeleton->m_owner->baseForm, TESForm, TESFullName);
			if (bname)
				return bname->GetName();
		}
		return "";
	}

	void ActorManager::Skeleton::addArmor(NiNode* armorModel)
	{
		IDType id = armors.size() ? armors.back().id + 1 : 0;
		auto prefix = armorPrefix(id);
		// FIXME we probably could simplify this by using findNode as surely we don't merge Armors with lurkers skeleton?
		npc = getNpcNode(skeleton);
		auto physicsFile = DefaultBBP::instance()->scanBBP(armorModel);

		armors.push_back(Armor());
		armors.back().id = id;
		armors.back().prefix = prefix;
		armors.back().physicsFile = physicsFile;

		doSkeletonMerge(npc, armorModel, prefix, armors.back().renameMap);
	}

	void ActorManager::Skeleton::attachArmor(NiNode* armorModel, NiAVObject* attachedNode)
	{
#ifdef _DEBUG
		if (armors.size() == 0 || armors.back().hasPhysics())
			_MESSAGE("Not attaching armor - no record or physics already exists");
#endif // _DEBUG

		Armor& armor = armors.back();
		armor.armorWorn = attachedNode;

		if (!isFirstPersonSkeleton(skeleton))
		{
			std::unordered_map<IDStr, IDStr> renameMap = armor.renameMap;
			// FIXME we probably could simplify this by using findNode as surely we don't attach Armors to lurkers skeleton?
			auto system = SkyrimSystemCreator().createSystem(getNpcNode(skeleton), attachedNode, armor.physicsFile, std::move(renameMap));

			if (system)
			{
				armor.setPhysics(system, isActive);
				hasPhysics = true;
			}
		}
	}

	void ActorManager::Skeleton::cleanArmor()
	{
		for (auto& i : armors)
		{
			if (!i.armorWorn) continue;
			if (i.armorWorn->m_parent) continue;

			i.clearPhysics();
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
#ifdef _DEBUG
				if (cleanAll)
					_DMESSAGE("cleaning headpart %s due to clean all", headPart.headPart->m_name);
				else
					_DMESSAGE("headpart %s disconnected", headPart.headPart->m_name);
#endif // _DEBUG

				auto renameIt = this->head.renameMap.begin();

				while (renameIt != this->head.renameMap.end())
				{
					bool erase = false;

					if (headPart.renamedBonesInUse.count(renameIt->first) != 0)
					{
						auto findNode = this->head.nodeUseCount.find(renameIt->first);
						if (findNode != this->head.nodeUseCount.end())
						{
							findNode->second -= 1;
#ifdef _DEBUG
							_DMESSAGE("decrementing use count by 1, it is now %d", findNode->second);
#endif // _DEBUG
							if (findNode->second <= 0)
							{
#ifdef _DEBUG
								_DMESSAGE("node no longer in use, cleaning from skeleton");
#endif // _DEBUG
								auto removeObj = findObject(npc, renameIt->second->cstr());
								if (removeObj)
								{
#ifdef _DEBUG
									_DMESSAGE("found node %s, removing", removeObj->m_name);
#endif // _DEBUG
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
				headPart.clearPhysics();
				headPart.renamedBonesInUse.clear();
			}
		}

		head.headParts.erase(std::remove_if(head.headParts.begin(), head.headParts.end(),
			[](Head::HeadPart& i) { return !i.headPart; }), head.headParts.end());
	}

	void ActorManager::Skeleton::clear()
	{
		std::for_each(armors.begin(), armors.end(), [](Armor& armor) { armor.clearPhysics(); });
		SkyrimPhysicsWorld::get()->removeSystemByNode(npc);
		cleanHead();
		head.headParts.clear();
		head.headNode = nullptr;
		armors.clear();
	}

	// Is called to print messages only
	bool ActorManager::Skeleton::checkPhysics()
	{
		hasPhysics = false;
		std::for_each(armors.begin(), armors.end(), [=](Armor& armor) {
			if (armor.state() != ItemState::e_NoPhysics)
				hasPhysics = true;
			});
		if (!hasPhysics)
			std::for_each(head.headParts.begin(), head.headParts.end(), [=](Head::HeadPart& headPart) {
			if (headPart.state() != ItemState::e_NoPhysics)
				hasPhysics = true;
				});
		_MESSAGE("%s isDrawn %d: %d", name(), hasPhysics);

		return hasPhysics;
	}

	bool ActorManager::Skeleton::isActiveInScene() const
	{
		// TODO: do this better
		// When entering/exiting an interior, NPCs are detached from the scene but not unloaded, so we need to check two levels up.
		// This properly removes exterior cell armors from the physics world when entering an interior, and vice versa.
		return skeleton->m_parent && skeleton->m_parent->m_parent && skeleton->m_parent->m_parent->m_parent;
	}

	bool ActorManager::Skeleton::isPlayerCharacter() const
	{
		// TODO magic number to remove
		return skeletonOwner == *g_thePlayer.GetPtr() || (skeleton->m_owner && skeleton->m_owner->formID == 0x14);
	}

	std::optional<NiPoint3> ActorManager::Skeleton::position() const
	{
		if (npc)
		{
			// This works for lurker skeletons.
			auto rootNode = findNode(npc, "NPC Root [Root]");
			if (rootNode) return rootNode->m_worldTransform.pos;
		}
		return std::optional<NiPoint3>();
	}

	void ActorManager::Skeleton::updateAttachedState(NiPoint3 cameraPosition, float maxDistance2, const NiNode* playerCell, NiPoint3 cameraOrientation, float maxAngleCosinus2)
	{
		// 1- Skeletons that aren't active in any scene are always detached, unless they are in the
		// same cell as the player character (workaround for issue in Ancestor Glade).
		// 2- Player character is always attached.
		// 3- Otherwise, attach only if both the camera and this skeleton have a position,
		// the distance between them is below the threshold value,
		// and the angle difference between the camera orientation and the skeleton orientation is below the threshold value.
		isActive = false;
		state = SkeletonState::e_InactiveNotInScene;

		if (isActiveInScene() || skeleton->m_parent && skeleton->m_parent->m_parent == playerCell)
		{
			if (isPlayerCharacter())
			{
				isActive = true;
				state = SkeletonState::e_ActiveIsPlayer;
			}
			else
			{
				state = SkeletonState::e_InactiveTooFar;
				auto pos = position();
				if (pos.has_value())
				{
					// We calculate the vector between camera and the skeleton feets.
					auto camera2SkeletonVector = pos.value() - cameraPosition;
					// This is the distance (squared) between the camera and the skeleton feets.
					auto c2SVMagnitude2 = camera2SkeletonVector.x * camera2SkeletonVector.x + camera2SkeletonVector.y * camera2SkeletonVector.y + camera2SkeletonVector.z * camera2SkeletonVector.z;

					/* If the distance is greater than the max distance, we let the skeleton inactive.
					 * We compare their squares rather than the distances themselves to avoid costly sqrt() operations. */
					if (c2SVMagnitude2 < maxDistance2)
					{
						// TODO Precision: rather than working with the position of the skeleton feets, we could work with the skeleton size.
						/* We calculate the angle between the camera direction and the camera2SkeletonVector.
						 * If the angle is lower than the max angle, we make the skeleton active.
						 * Demonstration leading to our check, based on scalar product of vectors (sometimes called inner product):
						 * 1/ dot(v1->, v2->) = |v1->| * |v2->| * cos(a) with a the angle between v1-> and v2->
						 * 2/ dot(v1->, v2->) = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
						 * => cos(a) = (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / (|v1->| * |v2->|)
						 * => cos^2(a) = (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z)^2 / ((v1.x^2 + v1.y^2 + v1.z^2) * (v2.x^2 + v2.y^2 + v2.z^2))
						 * In our case, we have:
						 * 1/ a < maxAngle (our test)
						 * 2/ |cameraOrientation| = 1 (because I built it by applying the rotational matrix to (0,1,0) which length is 1.)
						 * => cos(a) > cos(maxAngle)
						 * => cos^2(a) > cos^2(maxAngle) && cos(a) > 0
						 * => (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z)^2 / ((v1.x^2 + v1.y^2 + v1.z^2)) > cos^2(maxAngle)
						 *    with v1-> = camera2SkeletonVector and v2-> = cameraOrientation
						 *    && dot(v1->, v2->) > 0
						 * => a^2 / c2SVMagnitude2 > cos^2(maxAngle)
						 *	  && a > 0
						 *    with a = dot(camera2SkeletonVector, cameraOrientation)
						 *           = camera2SkeletonVector.x * cameraOrientation.x + camera2SkeletonVector.y * cameraOrientation.y + camera2SkeletonVector.z * cameraOrientation.z
						 * => a > 0 && a^2 > cos^2(maxAngle) * c2SVMagnitude2
						 *    with a = camera2SkeletonVector.x * cameraOrientation.x + camera2SkeletonVector.y * cameraOrientation.y + camera2SkeletonVector.z * cameraOrientation.z
						 */
						auto a = camera2SkeletonVector.x * cameraOrientation.x + camera2SkeletonVector.y * cameraOrientation.y + camera2SkeletonVector.z * cameraOrientation.z;
						if (a > 0 && a * a > maxAngleCosinus2 * c2SVMagnitude2)
						{
							isActive = true;
							state = SkeletonState::e_ActiveNearPlayer;
						}
					}
				}
			}
		}

		// We update the active state of armors and head parts.
		std::for_each(armors.begin(), armors.end(), [=](Armor& armor) { armor.updateActive(isActive); });
		std::for_each(head.headParts.begin(), head.headParts.end(), [=](Head::HeadPart& headPart) { headPart.updateActive(isActive); });
	}

	void ActorManager::Skeleton::reloadMeshes()
	{
		for (auto& i : armors)
		{
			i.clearPhysics();

			if (!isFirstPersonSkeleton(skeleton))
			{
				std::unordered_map<IDStr, IDStr> renameMap = i.renameMap;

				auto system = SkyrimSystemCreator().createSystem(npc, i.armorWorn, i.physicsFile, std::move(renameMap));

				if (system)
				{
					i.setPhysics(system, isActive);
					hasPhysics = true;
				}
			}
		}
		scanHead();
	}

	void ActorManager::Skeleton::scanHead()
	{
		if (isFirstPersonSkeleton(this->skeleton))
		{
#ifdef _DEBUG
			_DMESSAGE("not scanning head of first person skeleton");
#endif // _DEBUG
			return;
		}

		if (!this->head.headNode)
		{
#ifdef _DEBUG
			_DMESSAGE("actor has no head node");
#endif // _DEBUG
			return;
		}

		std::unordered_set<std::string> physicsDupes;

		for (auto& headPart : this->head.headParts)
		{
			// always regen physics for all head parts
			headPart.clearPhysics();

			if (headPart.physicsFile.first.empty())
			{
#ifdef _DEBUG
				_DMESSAGE("no physics file for headpart %s", headPart.headPart->m_name);
#endif // _DEBUG
				continue;
			}

			if (physicsDupes.count(headPart.physicsFile.first))
			{
#ifdef _DEBUG
				_DMESSAGE("previous head part generated physics system for file %s, skipping",
					headPart.physicsFile.first.c_str());
#endif // _DEBUG
				continue;
			}

			std::unordered_map<IDStr, IDStr> renameMap = this->head.renameMap;

#ifdef _DEBUG
			_DMESSAGE("try create system for headpart %s physics file %s", headPart.headPart->m_name,
				headPart.physicsFile.first.c_str());
#endif // _DEBUG
			physicsDupes.insert(headPart.physicsFile.first);
			auto system = SkyrimSystemCreator().createSystem(npc, this->head.headNode, headPart.physicsFile,
				std::move(renameMap));

			if (system)
			{
#ifdef _DEBUG
				_DMESSAGE("success");
#endif // _DEBUG
				headPart.setPhysics(system, isActive);
				hasPhysics = true;
			}
		}
	}

	typedef bool (*_TESNPC_GetFaceGeomPath)(TESNPC* a_npc, char* a_buf);
	RelocAddr<_TESNPC_GetFaceGeomPath> TESNPC_GetFaceGeomPath(offset::TESNPC_GetFaceGeomPath);

	void ActorManager::Skeleton::processGeometry(BSFaceGenNiNode* headNode, BSGeometry* geometry)
	{
		if (this->head.headNode && this->head.headNode != headNode)
		{
#ifdef _DEBUG
			_DMESSAGE("completely new head attached to skeleton, clearing tracking");
#endif // _DEBUG
			for (auto& headPart : this->head.headParts)
			{
				headPart.clearPhysics();
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
		++this->head.id;
		this->head.prefix = headPrefix(this->head.id);

		auto it = std::find_if(this->head.headParts.begin(), this->head.headParts.end(),
			[geometry](const Head::HeadPart& p)
			{
				return p.headPart == geometry;
			});

		if (it != this->head.headParts.end())
		{
#ifdef _DEBUG
			_DMESSAGE("geometry is already added as head part");
#endif // _DEBUG
			return;
		}

		this->head.headParts.push_back(Head::HeadPart());

		head.headParts.back().headPart = geometry;
		head.headParts.back().clearPhysics();

		// Skinning
#ifdef _DEBUG
		_DMESSAGE("skinning geometry to skeleton");
#endif // _DEBUG

		if (!geometry->m_spSkinInstance || !geometry->m_spSkinInstance->m_spSkinData)
		{
			_ERROR("geometry is missing skin instance - how?");
			return;
		}

		auto fmd = static_cast<BSFaceGenModelExtraData*>(geometry->GetExtraData("FMD"));

		BSGeometry* origGeom = nullptr;
		NiGeometry* origNiGeom = nullptr;

		if (fmd && fmd->m_model && fmd->m_model->unk10 && fmd->m_model->unk10->unk08)
		{
#ifdef _DEBUG
			_DMESSAGE("orig part node found via fmd");
#endif // _DEBUG
			auto origRootNode = fmd->m_model->unk10->unk08->GetAsNiNode();
			head.headParts.back().physicsFile = DefaultBBP::instance()->scanBBP(origRootNode);
			head.headParts.back().origPartRootNode = origRootNode;
			for (int i = 0; i < origRootNode->m_children.m_size; i++)
			{
				if (origRootNode->m_children.m_data[i])
				{
					const auto geo = origRootNode->m_children.m_data[i]->GetAsBSGeometry();

					if (geo)
					{
						origGeom = geo;
						break;
					}
				}
			}
		}
		else
		{
#ifdef _DEBUG
			_DMESSAGE("no fmd available, loading original facegeom");
#endif // _DEBUG
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
#ifdef _DEBUG
							_DMESSAGE("loading facegeom from path %s", filePath);
#endif // _DEBUG
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
#ifdef _DEBUG
										_DMESSAGE("npc root fadenode found");
#endif // _DEBUG
										head.npcFaceGeomNode = rootFadeNode;
									}
#ifdef _DEBUG
									else
									{
										_DMESSAGE("npc facegeom root wasn't fadenode as expected");
									}
#endif // _DEBUG

								}
								CALL_MEMBER_FN(niStream, dtor)();
							}
						}
					}
				}
			}
#ifdef _DEBUG
			else
			{
				_DMESSAGE("using cached facegeom");
			}
#endif // _DEBUG
			if (head.npcFaceGeomNode)
			{
				head.headParts.back().physicsFile = DefaultBBP::instance()->scanBBP(head.npcFaceGeomNode);
				auto obj = findObject(head.npcFaceGeomNode, geometry->m_name);
				if (obj)
				{
					auto ob = obj->GetAsBSGeometry();
					if (ob) origGeom = ob;
					else {
						auto on = obj->GetAsNiGeometry();
						if (on) origNiGeom = on;
					}
				}
			}
		}

		bool hasMerged = false;
		bool hasRenames = false;

		for (int boneIdx = 0; boneIdx < geometry->m_spSkinInstance->m_spSkinData->m_uiBones; boneIdx++)
		{
			BSFixedString boneName("");

			// skin the way the game does via FMD
			if (boneIdx <= 7)
			{
				if (fmd)
					boneName = fmd->bones[boneIdx];
			}

			if (!*boneName.c_str())
			{
				if (origGeom)
				{
					boneName = origGeom->m_spSkinInstance->m_ppkBones[boneIdx]->m_name;
				}
				else if (origNiGeom)
				{
					boneName = origNiGeom->m_spSkinInstance->m_ppkBones[boneIdx]->m_name;
				}
			}

			auto renameIt = this->head.renameMap.find(boneName.c_str());

			if (renameIt != this->head.renameMap.end())
			{
#ifdef _DEBUG
				_DMESSAGE("found renamed bone %s -> %s", boneName, renameIt->second->cstr());
#endif // _DEBUG
				boneName = renameIt->second->cstr();
				hasRenames = true;
			}

			auto boneNode = findNode(this->npc, boneName);

			if (!boneNode && !hasMerged)
			{
#ifdef _DEBUG
				_DMESSAGE("bone not found on skeleton, trying skeleton merge");
#endif // _DEBUG
				if (this->head.headParts.back().origPartRootNode)
				{
					doSkeletonMerge(npc, head.headParts.back().origPartRootNode, head.prefix, head.renameMap);
				}
				else if (this->head.npcFaceGeomNode)
				{
					// Facegen data doesn't have any tree structure to the skeleton. We need to make any new
					// nodes children of the head node, so that they move properly when there's no physics.
					// This case never happens to a lurker skeleton, thus we don't need to test.
					auto headNode = findNode(head.npcFaceGeomNode, "NPC Head [Head]");
					if (headNode)
					{
						NiTransform invTransform;
						headNode->m_localTransform.Invert(invTransform);
						for (int i = 0; i < head.npcFaceGeomNode->m_children.m_arrayBufLen; ++i)
						{
							Ref<NiNode> child = castNiNode(head.npcFaceGeomNode->m_children.m_data[i]);
							// This case never happens to a lurker skeleton, thus we don't need to test.
							if (child && !findNode(npc, child->m_name))
							{
								child->m_localTransform = invTransform * child->m_localTransform;
								head.npcFaceGeomNode->RemoveAt(i);
								headNode->AttachChild(child, false);
							}
						}
					}
					doSkeletonMerge(npc, this->head.npcFaceGeomNode, head.prefix, head.renameMap);
				}
				hasMerged = true;

				auto postMergeRenameIt = this->head.renameMap.find(boneName.c_str());

				if (postMergeRenameIt != this->head.renameMap.end())
				{
#ifdef _DEBUG
					_DMESSAGE("found renamed bone %s -> %s", boneName, postMergeRenameIt->second->cstr());
#endif // _DEBUG
					boneName = postMergeRenameIt->second->cstr();
					hasRenames = true;
				}

			}

			if (!boneNode)
			{
				_ERROR("bone %s not found after skeleton merge, geometry cannot be fully skinned", boneName);
				continue;
			}

			geometry->m_spSkinInstance->m_ppkBones[boneIdx] = boneNode;
			geometry->m_spSkinInstance->m_worldTransforms[boneIdx] = &boneNode->m_worldTransform;
		}

		geometry->m_spSkinInstance->m_pkRootParent = headNode;

		if (hasRenames)
		{
			for (auto& entry : head.renameMap)
			{
				if ((this->head.headParts.back().origPartRootNode && findObject(this->head.headParts.back().origPartRootNode, entry.first->cstr())) ||
					(this->head.npcFaceGeomNode && findObject(this->head.npcFaceGeomNode, entry.first->cstr())))
				{
					auto findNode = this->head.nodeUseCount.find(entry.first);
					if (findNode != this->head.nodeUseCount.end())
					{
						findNode->second += 1;
#ifdef _DEBUG
						_DMESSAGE("incrementing use count by 1, it is now %d", findNode->second);
#endif // _DEBUG
					}
					else
					{
						this->head.nodeUseCount.insert(std::make_pair(entry.first, 1));
#ifdef _DEBUG
						_DMESSAGE("first use of bone, count 1");
#endif // _DEBUG
					}
					head.headParts.back().renamedBonesInUse.insert(entry.first);
				}
			}
		}

#ifdef _DEBUG
		_DMESSAGE("done skinning part");
#endif // _DEBUG
	}
}
