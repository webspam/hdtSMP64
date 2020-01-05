#pragma once

#include "hdtSkyrimSystem.h"
#include "hdtSkinnedMesh\hdtSkinnedMeshWorld.h"
#include "IEventListener.h"
#include "HookEvents.h"

#include <atomic>
#include "ActorManager.h"
#include "skse64/PapyrusEvents.h"

namespace hdt
{
	constexpr float RESET_PHYSICS = -10.0f;

	class SkyrimPhysicsWorld : public SkinnedMeshWorld, public IEventListener<FrameEvent>, public IEventListener<ShutdownEvent>, public BSTEventSink<SKSECameraEvent>
	{
	public:

		static SkyrimPhysicsWorld* get();

		void doUpdate(float delta);
		void updateActiveState();

		virtual void addSkinnedMeshSystem(SkinnedMeshSystem* system) override;
		virtual void removeSkinnedMeshSystem(SkinnedMeshSystem* system) override;
		void removeSystemByNode(void* root);

		void resetTransformsToOriginal();
		void resetSystems();

		virtual void onEvent(const FrameEvent& e) override;
		virtual void onEvent(const ShutdownEvent& e) override;

		virtual	EventResult		ReceiveEvent(SKSECameraEvent* evn, EventDispatcher<SKSECameraEvent>* dispatcher) override;

		inline bool isSuspended() { return m_suspended; }
		inline void suspend(bool loading = false) { m_suspended = true; m_loading = loading; }
		inline void resume() {
			m_suspended = false;
			if (m_loading)
			{
				ActorManager::instance()->reloadMeshes();
				resetSystems();
				m_loading = false;
			}
		}

		btVector3 applyTranslationOffset();
		void restoreTranslationOffset(const btVector3&);

		float m_timeTick = 1 / 60.f;
		bool m_clampRotations = true;
		bool m_unclampedResets = true;
		float m_unclampedResetAngle = 120.0f;
		uint8_t m_resetPc;

	private:

		SkyrimPhysicsWorld(void);
		~SkyrimPhysicsWorld(void);

		std::mutex m_lock;

		std::atomic_bool m_suspended;
		std::atomic_bool m_loading;
		float m_averageInterval;
		float m_accumulatedInterval;
	};
}
