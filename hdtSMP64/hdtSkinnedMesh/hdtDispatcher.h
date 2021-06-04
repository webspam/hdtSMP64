#pragma once

#include "hdtBulletHelper.h"
#include <ppl.h>
#include <ppltasks.h>
#include <vector>

namespace hdt
{
	class SkinnedMeshBody;

	class CollisionDispatcher : public btCollisionDispatcher
	{
	public:

		CollisionDispatcher(btCollisionConfiguration* collisionConfiguration) : btCollisionDispatcher(
			collisionConfiguration)
		{
		}

		btPersistentManifold* getNewManifold(const btCollisionObject* b0, const btCollisionObject* b1) override
		{
			m_lock.lock();
			auto ret = btCollisionDispatcher::getNewManifold(b0, b1);
			m_lock.unlock();
			return ret;
		}

		void releaseManifold(btPersistentManifold* manifold) override
		{
			m_lock.lock();
			btCollisionDispatcher::releaseManifold(manifold);
			m_lock.unlock();
		}

		bool needsCollision(const btCollisionObject* body0, const btCollisionObject* body1) override;
		void dispatchAllCollisionPairs(btOverlappingPairCache* pairCache, const btDispatcherInfo& dispatchInfo,
		                               btDispatcher* dispatcher) override;

		int getNumManifolds() const override;
		btPersistentManifold** getInternalManifoldPointer() override;
		btPersistentManifold* getManifoldByIndexInternal(int index) override;

		void clearAllManifold();

		std::mutex m_lock;
		std::vector<std::pair<SkinnedMeshBody*, SkinnedMeshBody*>> m_pairs;
		std::vector<std::function<void()>> m_immediateFuncs;
		std::vector<std::function<void()>> m_delayedFuncs;
	};
}
