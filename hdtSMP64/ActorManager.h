#pragma once

#include "../hdtSSEUtils/NetImmerseUtils.h"
#include "../hdtSSEUtils/FrameworkUtils.h"

#include "hdtSkyrimSystem.h"

#include "IEventListener.h"
#include "HookEvents.h"

#include <mutex>
#include <optional>

namespace hdt
{
	class ActorManager
		: public IEventListener<ArmorAttachEvent>
		  , public IEventListener<SkinSingleHeadGeometryEvent>
		  , public IEventListener<SkinAllHeadGeometryEvent>
		  , public IEventListener<FrameEvent>
		  , public IEventListener<ShutdownEvent>
	{
	public:

		enum PhysicsState
		{
			e_NoPhysics,
			e_Inactive,
			e_Active
		};

	private:
		struct Skeleton;

		struct PhysicsItem
		{
			std::string physicsFile;

			void setPhysics(Ref<SkyrimSystem>& system, bool active);
			void clearPhysics();
			PhysicsState state() const;

			const std::vector<Ref<SkinnedMeshBody>>& meshes() const;

			void updateActive(bool active);
		private:
			Ref<SkyrimSystem> m_physics;
		};

		struct Head
		{
			struct HeadPart : public PhysicsItem
			{
				Ref<BSGeometry> headPart;
				Ref<NiNode> origPartRootNode;
				std::string physicsFile;
				std::set<IDStr> renamedBonesInUse;
			};

			Ref<IString> prefix;
			Ref<BSFaceGenNiNode> headNode;
			Ref<BSFadeNode> npcFaceGeomNode;
			std::vector<HeadPart> headParts;
			std::unordered_map<IDStr, IDStr> renameMap;
			std::unordered_map<IDStr, uint8_t> nodeUseCount;
			bool isFullSkinning;
		};

		struct Armor : public PhysicsItem
		{
			Ref<IString> prefix;
			Ref<NiAVObject> armorWorn;
			std::unordered_map<IDStr, IDStr> renameMap;
		};

		struct Skeleton
		{
			NiPointer<TESObjectREFR> skeletonOwner;
			Ref<NiNode> skeleton;
			Ref<NiNode> npc;
			Head head;

			void addArmor(NiNode* armorModel);
			void attachArmor(NiNode* armorModel, NiAVObject* attachedNode);

			void cleanArmor();
			void cleanHead(bool cleanAll = false);
			void clear();

			bool isActiveInScene() const;
			bool isPlayerCharacter() const;
			std::optional<NiPoint3> position() const;

			void updateAttachedState(std::optional<NiPoint3> playerPosition, float maxDistance, const NiNode* playerCell);
			void reloadMeshes();

			void scanHead();
			void processGeometry(BSFaceGenNiNode* head, BSGeometry* geometry);

			static void doSkeletonMerge(NiNode* dst, NiNode* src, IString* prefix,
			                            std::unordered_map<IDStr, IDStr>& map);
			static void doSkeletonClean(NiNode* dst, IString* prefix);
			static NiNode* cloneNodeTree(NiNode* src, IString* prefix, std::unordered_map<IDStr, IDStr>& map);
			static void renameTree(NiNode* root, IString* prefix, std::unordered_map<IDStr, IDStr>& map);

			// TODO: refactor this is just to get it working for now
			static void doHeadSkeletonMerge(NiNode* dst, NiNode* src, IString* prefix,
			                                std::unordered_map<IDStr, IDStr>& map);
			static void renameHeadTree(NiNode* root, IString* prefix, std::unordered_map<IDStr, IDStr>& map);
			static NiNode* cloneHeadNodeTree(NiNode* src, IString* prefix, std::unordered_map<IDStr, IDStr>& map);

			const std::vector<Armor>& getArmors() { return armors; }

		private:
			bool isActive = false;
			std::vector<Armor> armors;
		};

		bool m_shutdown = false;
		std::recursive_mutex m_lock;
		std::vector<Skeleton> m_skeletons;



		Skeleton& getSkeletonData(NiNode* skeleton);

	public:
		ActorManager();
		~ActorManager();

		static ActorManager* instance();
		static IDStr generatePrefix(NiAVObject* armor);

		void onEvent(const ArmorAttachEvent& e) override;
		void onEvent(const FrameEvent& e) override;
		void onEvent(const ShutdownEvent&) override;
		void onEvent(const SkinSingleHeadGeometryEvent&) override;
		void onEvent(const SkinAllHeadGeometryEvent&) override;

		void reloadMeshes();

		std::vector<Skeleton> getSkeletons() const;

		bool m_skinNPCFaceParts = true;
		float m_maxDistance = 1e4f;
	};
}
