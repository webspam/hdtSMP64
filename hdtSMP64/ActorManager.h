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
		using IDType = UInt32;

	public:

		enum ItemState
		{
			e_NoPhysics,
			e_Inactive,
			e_Active
		};

		// Overall skeleton state, purely for console debug info
		enum SkeletonState
		{
			// Note order: inactive states must come before e_SkeletonActive, and active states after
			e_InactiveNotInScene,
			e_InactiveTooFar,
			e_SkeletonActive,
			e_ActiveNearPlayer,
			e_ActiveIsPlayer
		};

	private:
		struct Skeleton;

		struct PhysicsItem
		{
			DefaultBBP::PhysicsFile physicsFile;

			void setPhysics(Ref<SkyrimSystem>& system, bool active);
			void clearPhysics();
			bool hasPhysics() const { return m_physics; }
			ItemState state() const;

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
				std::set<IDStr> renamedBonesInUse;
			};

			IDType id;
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
			IDType id;
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
			SkeletonState state;

			std::string name();
			void addArmor(NiNode* armorModel);
			void attachArmor(NiNode* armorModel, NiAVObject* attachedNode);

			void cleanArmor();
			void cleanHead(bool cleanAll = false);
			void clear();

			bool isPlayerCharacter() const;
			bool hasPhysics = false;
			std::optional<NiPoint3> position() const;

			void updateAttachedState(std::optional<NiPoint3> playerPosition, float maxDistance, const NiNode* playerCell, std::optional<NiPoint3> playerRotation, float maxAngle);
			void reloadMeshes();

			void scanHead();
			void processGeometry(BSFaceGenNiNode* head, BSGeometry* geometry);

			static void doSkeletonMerge(NiNode* dst, NiNode* src, IString* prefix,
			                            std::unordered_map<IDStr, IDStr>& map);
			static void doSkeletonClean(NiNode* dst, IString* prefix);
			static NiNode* cloneNodeTree(NiNode* src, IString* prefix, std::unordered_map<IDStr, IDStr>& map);
			static void renameTree(NiNode* root, IString* prefix, std::unordered_map<IDStr, IDStr>& map);

			const std::vector<Armor>& getArmors() { return armors; }

		private:
			bool isActiveInScene() const;
			bool checkPhysics();
			int headMeshes = 0;
			int armorMeshes = 0;

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

		static IDStr armorPrefix(IDType id);
		static IDStr headPrefix(IDType id);

		void onEvent(const ArmorAttachEvent& e) override;
		void onEvent(const FrameEvent& e) override;
		void onEvent(const ShutdownEvent&) override;
		void onEvent(const SkinSingleHeadGeometryEvent&) override;
		void onEvent(const SkinAllHeadGeometryEvent&) override;

		void reloadMeshes();

		std::vector<Skeleton> getSkeletons() const;

		bool m_skinNPCFaceParts = true;
		float m_maxDistance = 1e4f;
		float m_maxAngle = 45.0;
	};
}
