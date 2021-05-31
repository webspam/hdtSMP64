#include "hdtDispatcher.h"
#include "hdtSkinnedMeshBody.h"
#include "hdtSkinnedMeshAlgorithm.h"
#include "hdtCudaInterface.h"
#include "hdtFrameTimer.h"

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

		bool haveCuda = CudaInterface::instance()->hasCuda() && (!FrameTimer::instance()->running() || FrameTimer::instance()->cudaFrame());

		FrameTimer::instance()->logEvent(FrameTimer::e_Start);

		if (haveCuda)
		{
			std::vector<SkinnedMeshBody*> bodies;
			bodies.reserve(to_update.size());
			std::vector<PerVertexShape*> vertexShapes;
			vertexShapes.reserve(to_update.size());
			std::vector<PerTriangleShape*> triangleShapes;
			triangleShapes.reserve(to_update.size());
			bool initialized = true;

			// Build simple vectors of the things to update, and determine whether any new CUDA objects need
			// to be created
			for (auto& o : to_update)
			{
				bodies.push_back(o.first);
				initialized &= static_cast<bool>(bodies.back()->m_cudaObject);
				if (o.second.first)
				{
					vertexShapes.push_back(o.second.first);
					initialized &= static_cast<bool>(vertexShapes.back()->m_cudaObject);
				}
				if (o.second.second)
				{
					triangleShapes.push_back(o.second.second);
					initialized &= static_cast<bool>(triangleShapes.back()->m_cudaObject);
				}
			}

			// Create any new CUDA objects if necessary
			if (!initialized)
			{
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
			}

			// Update bone transforms and launch the vertex calculation kernel
			for (auto body : bodies)
			{
				body->updateBones();
				body->m_cudaObject->launch();
			}

			// Launch per-triangle kernels. Theoretically we should get better performance launching kernels
			// breadth-first like this.
			for (auto triangleShape : triangleShapes)
			{
				triangleShape->m_cudaObject->launch();
			}
			for (auto triangleShape : triangleShapes)
			{
				triangleShape->m_cudaObject->launchTree();
			}

			// Launch per-vertex kernels
			for (auto vertexShape : vertexShapes)
			{
				vertexShape->m_cudaObject->launch();
			}
			for (auto vertexShape : vertexShapes)
			{
				vertexShape->m_cudaObject->launchTree();
			}

			// Update the aggregate parts of the AABB trees
			for (auto& o : to_update)
			{
				o.first->m_cudaObject->synchronize();

				if (o.second.first)
				{
					o.second.first->m_cudaObject->updateTree();
				}
				if (o.second.second)
				{
					o.second.second->m_cudaObject->updateTree();
				}
				o.first->m_bulletShape.m_aabb = o.first->m_shape->m_tree.aabbAll;
			}
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

		FrameTimer::instance()->logEvent(FrameTimer::e_Internal);

		if (haveCuda)
		{
			CudaInterface::instance()->clearBufferPool();

			// Launch collision checking
			std::vector<std::function<void()>> collisionFuncs(m_pairs.size());
			for (int i = 0; i < m_pairs.size(); ++i)
			{
				auto& pair = m_pairs[i];
				if (pair.first->m_shape->m_tree.collapseCollideL(&pair.second->m_shape->m_tree))
				{
					SkinnedMeshAlgorithm::queueCollision(collisionFuncs.begin() + i, pair.first, pair.second, this);
				}
			}

			// Synchronize and apply the collision results
			concurrency::parallel_for_each(collisionFuncs.begin(), collisionFuncs.end(),
				[](std::function<void()>& f)
			{
				if (f)
					f();
			});
		}
		else
		{
			// Now we can process the collisions
			concurrency::parallel_for_each(m_pairs.begin(), m_pairs.end(),
				[this](std::pair<SkinnedMeshBody*, SkinnedMeshBody*>& i)
			{
				if (i.first->m_shape->m_tree.collapseCollideL(&i.second->m_shape->m_tree))
				{
					SkinnedMeshAlgorithm::processCollision(i.first, i.second, this);
				}
			});
		}

		m_pairs.clear();

		FrameTimer::instance()->logEvent(FrameTimer::e_End);
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
