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

					auto it0 = to_update.insert({ shape0, {nullptr, nullptr} });
					auto it1 = to_update.insert({ shape1, {nullptr, nullptr} });

					m_pairs.push_back(std::make_pair(shape0, shape1));

					auto a = shape0->m_shape->asPerTriangleShape();
					auto b = shape1->m_shape->asPerTriangleShape();
					if (a)
						it0.first->second.second = a;
					else
						it0.first->second.first = shape0->m_shape->asPerVertexShape();
					if (b)
						it1.first->second.second = b;
					else
						it1.first->second.first = shape1->m_shape->asPerVertexShape();
					if (a && b)
					{
						it0.first->second.first = a->m_verticesCollision;
						it1.first->second.first = b->m_verticesCollision;
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

			// Launch per-triangle kernels. Performance is significantly better grouping kernels by type like this
			// instead of setting up each complete stream in turn - presumably because launch overhead is reduced.
			std::for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				if (o.second.second)
				{
					o.second.second->m_cudaObject->launch();
				}
			});

			// Launch per-vertex kernels and vertex transfers. The transfer isn't a kernel launch so there shouldn't
			// be any cost to interleaving these.
			std::for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				if (o.second.first)
				{
					o.second.first->m_cudaObject->launch();
				}
				o.first->m_cudaObject->launchTransfer();
			});

			// Do the sequential part of the AABB tree updates.
			// TODO: Would like to have this concurrent with the actual collision processing, but currently
			// we have to wait for all the tree updates before any collision can be done.
			concurrency::parallel_for_each(to_update.begin(), to_update.end(), [](UpdateMap::value_type& o)
			{
				// Synchronize just the stream for this body (this is why we wanted bodies and meshes grouped
				// together). This will also trigger transfer of vertex data back to the host. We may be able
				// to get rid of that completely if we move the main collision detection algorithm to GPU.
				o.first->m_cudaObject->waitForAaabData();

				if (o.second.first)
				{
					o.second.first->m_tree.updateAabb();
				}
				if (o.second.second)
				{
					o.second.second->m_tree.updateAabb();
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

		// Now we can process the collisions, synchronizing each pair with both its bodies just before processing.
		concurrency::parallel_for_each(m_pairs.begin(), m_pairs.end(),
			[this](std::pair<SkinnedMeshBody*, SkinnedMeshBody*>& i)
			{
				if (i.first->m_shape->m_tree.collapseCollideL(&i.second->m_shape->m_tree))
				{
					if (CudaInterface::instance()->hasCuda())
					{
						i.first->m_cudaObject->synchronize();
						i.second->m_cudaObject->synchronize();
					}
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
