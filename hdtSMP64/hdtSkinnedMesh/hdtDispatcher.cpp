#include "hdtDispatcher.h"
#include "hdtSkinnedMeshBody.h"
#include "hdtSkinnedMeshAlgorithm.h"
#include "hdtCudaInterface.h"

#include <LinearMath/btPoolAllocator.h>

namespace hdt
{
	void CollisionDispatcher::clearAllManifold()
	{
		std::lock_guard<decltype(m_lock)> l(m_lock);
		for (int i = 0; i < m_manifoldsPtr.size(); ++i)
		{
			auto manifold = m_manifoldsPtr[i];
			manifold->~btPersistentManifold();
			if (m_persistentManifoldPoolAllocator->validPtr(manifold))
				m_persistentManifoldPoolAllocator->freeMemory(manifold);
			else
				btAlignedFree(manifold);
		}
		m_manifoldsPtr.clear();
	}

	bool needsCollision(const SkinnedMeshBody* shape0, const SkinnedMeshBody* shape1)
	{
		if (!shape0 || !shape1 || shape0 == shape1)
			return false;

		if (shape0->m_isKinematic && shape1->m_isKinematic)
			return false;

		return shape0->canCollideWith(shape1) && shape1->canCollideWith(shape0);
	}

	bool CollisionDispatcher::needsCollision(const btCollisionObject* body0, const btCollisionObject* body1)
	{
		auto shape0 = dynamic_cast<const SkinnedMeshBody*>(body0);
		auto shape1 = dynamic_cast<const SkinnedMeshBody*>(body1);

		if (shape0 || shape1)
		{
			return hdt::needsCollision(shape0, shape1);
		}
		if (body0->isStaticOrKinematicObject() && body1->isStaticOrKinematicObject())
			return false;
		if (body0->checkCollideWith(body1) || body1->checkCollideWith(body0))
		{
			auto rb0 = static_cast<SkinnedMeshBone*>(body0->getUserPointer());
			auto rb1 = static_cast<SkinnedMeshBone*>(body0->getUserPointer());

			return rb0->canCollideWith(rb1) && rb1->canCollideWith(rb0);
		}
		else return false;
	}

	void CollisionDispatcher::dispatchAllCollisionPairs(btOverlappingPairCache* pairCache,
	                                                    const btDispatcherInfo& dispatchInfo, btDispatcher* dispatcher)
	{
		auto size = pairCache->getNumOverlappingPairs();
		if (!size) return;

		m_pairs.reserve(size);
		auto pairs = pairCache->getOverlappingPairArrayPtr();

		SpinLock lock;

		using UpdateMap = std::unordered_map<SkinnedMeshBody*, std::pair<PerVertexShape*, PerTriangleShape*> >;
		UpdateMap to_update;

		// Find bodies and meshes that need collision checking. We want to keep them together in a map so they can
		// be grouped by CUDA stream
		concurrency::parallel_for(0, size, [&](int i)
		{
			auto& pair = pairs[i];

			auto shape0 = dynamic_cast<SkinnedMeshBody*>(static_cast<btCollisionObject*>(pair.m_pProxy0->m_clientObject));
			auto shape1 = dynamic_cast<SkinnedMeshBody*>(static_cast<btCollisionObject*>(pair.m_pProxy1->m_clientObject));

			if (shape0 || shape1)
			{
				if (hdt::needsCollision(shape0, shape1) && shape0->isBoundingSphereCollided(shape1))
				{
					HDT_LOCK_GUARD(l, lock);

					auto it0 = to_update.insert({ shape0, {nullptr, nullptr} }).first;
					auto it1 = to_update.insert({ shape1, {nullptr, nullptr} }).first;

					m_pairs.push_back(std::make_pair(shape0, shape1));

					auto a = shape0->m_shape->asPerTriangleShape();
					auto b = shape1->m_shape->asPerTriangleShape();
					if (a)
						it0->second.second = a;
					else
						it0->second.first = shape0->m_shape->asPerVertexShape();
					if (b)
						it1->second.second = b;
					else
						it1->second.first = shape1->m_shape->asPerVertexShape();
					if (a && b)
					{
						it0->second.first = a->m_verticesCollision;
						it1->second.first = b->m_verticesCollision;
					}
				}
			}
			else getNearCallback()(pair, *this, dispatchInfo);
		});

		if (CudaInterface::instance()->hasCuda())
		{
			// First create any CUDA objects that don't exist already. Each body has its own stream, and per-vertex
			// and per-triangle updates for it will be launched in the same stream.
			concurrency::parallel_for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				if (!o.first->m_cudaObject)
				{
					o.first->m_cudaObject.reset(new CudaBody(o.first));
				}
				if (o.second.first && !o.second.first->m_cudaObject)
				{
					o.second.first->m_cudaObject.reset(new CudaPerVertexShape(o.second.first));
				}
				if (o.second.second && !o.second.second->m_cudaObject)
				{
					o.second.second->m_cudaObject.reset(new CudaPerTriangleShape(o.second.second));
				}
			});

			// Update bone transforms and launch the vertex calculation kernel as soon as the transforms are ready
			// for each body.
			concurrency::parallel_for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				o.first->updateBones();
				o.first->m_cudaObject->launch();
			});

			// Launch per-triangle kernels. Theoretically we should get better performance launching kernels
			// breadth-first like this.
			for (auto& o : to_update)
			{
				if (o.second.second)
				{
					o.second.second->m_cudaObject->launch();
				}
			}
			for (auto& o : to_update)
			{
				if (o.second.second)
				{
					o.second.second->m_cudaObject->launchTree();
				}
			}
			for (auto& o : to_update)
			{
				if (o.second.first)
				{
					o.second.first->m_cudaObject->launch();
				}
			}
			for (auto& o : to_update)
			{
				if (o.second.first)
				{
					o.second.first->m_cudaObject->launchTree();
				}
				o.first->m_cudaObject->recordState();
			}

			// Update the aggregate parts of the AABB trees
			concurrency::parallel_for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				o.first->m_cudaObject->waitForAaabData();

				if (o.second.first)
				{
					o.second.first->m_cudaObject->updateTree();
				}
				if (o.second.second)
				{
					o.second.second->m_cudaObject->updateTree();
				}
				o.first->m_bulletShape.m_aabb = o.first->m_shape->m_tree.aabbAll;
			});
		}
		else
		{
			concurrency::parallel_for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				o.first->internalUpdate();
				if (o.second.first)
				{
					o.second.first->internalUpdate();
				}
				if (o.second.second)
				{
					o.second.second->internalUpdate();
				}
				o.first->m_bulletShape.m_aabb = o.first->m_shape->m_tree.aabbAll;
			});
		}

		CudaInterface::instance()->clearBufferPool();

		// Now we can process the collisions
		concurrency::parallel_for_each(m_pairs.begin(), m_pairs.end(),
			[this](std::pair<SkinnedMeshBody*, SkinnedMeshBody*>& i)
			{
				if (i.first->m_shape->m_tree.collapseCollideL(&i.second->m_shape->m_tree))
				{
					SkinnedMeshAlgorithm::processCollision(i.first, i.second, this);
				}
			});

		m_pairs.clear();
	}

	int CollisionDispatcher::getNumManifolds() const
	{
		return m_manifoldsPtr.size();
	}

	btPersistentManifold* CollisionDispatcher::getManifoldByIndexInternal(int index)
	{
		return m_manifoldsPtr[index];
	}

	btPersistentManifold** CollisionDispatcher::getInternalManifoldPointer()
	{
		return btCollisionDispatcher::getInternalManifoldPointer();
	}
}
