#pragma once

#include "hdtSkinnedMeshShape.h"
#include "hdtDispatcher.h"
#include "hdtCudaInterface.h"

// Define this to do actual collision checking on GPU. This is currently slow and has very inconsistent
// framerate. If not defined, the GPU will still be used if available for vertex and bounding box
// calculations, but collision will be done on the CPU.
#define USE_GPU_COLLISION

namespace hdt
{
	class SkinnedMeshAlgorithm : public btCollisionAlgorithm
	{
	public:
		SkinnedMeshAlgorithm(const btCollisionAlgorithmConstructionInfo& ci);

		void processCollision(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap,
		                      const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut) override
		{
		}

		btScalar calculateTimeOfImpact(btCollisionObject* body0, btCollisionObject* body1,
		                               const btDispatcherInfo& dispatchInfo, btManifoldResult* resultOut) override
		{
			return 1;
		} // TOI cost too much
		void getAllContactManifolds(btManifoldArray& manifoldArray) override
		{
		}

		struct CreateFunc : public btCollisionAlgorithmCreateFunc
		{
			btCollisionAlgorithm* CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo& ci,
			                                               const btCollisionObjectWrapper* body0Wrap,
			                                               const btCollisionObjectWrapper* body1Wrap) override
			{
				void* mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(SkinnedMeshAlgorithm));
				return new(mem) SkinnedMeshAlgorithm(ci);
			}
		};

		static void registerAlgorithm(btCollisionDispatcher* dispatcher);

		static const int MaxCollisionCount = 256;

		static void queueCollision(
			std::vector<std::function<void()>>::iterator queue,
			SkinnedMeshBody* body0Wrap,
			SkinnedMeshBody* body1Wrap,
			CollisionDispatcher* dispatcher);

		static void processCollision(SkinnedMeshBody* body0Wrap, SkinnedMeshBody* body1Wrap,
		                             CollisionDispatcher* dispatcher);
	protected:

		struct CollisionMerge
		{
			btVector3 normal;
			btVector3 pos[2];
			float weight;

			CollisionMerge()
			{
				_mm_store_ps(((float*)this), _mm_setzero_ps());
				_mm_store_ps(((float*)this) + 4, _mm_setzero_ps());
				_mm_store_ps(((float*)this) + 8, _mm_setzero_ps());
				_mm_store_ps(((float*)this) + 12, _mm_setzero_ps());
			}
		};

		struct MergeBuffer
		{
			MergeBuffer()
			{
				mergeStride = mergeSize = 0;
				buffer = nullptr;
			}

			CollisionMerge* begin() const { return buffer; }
			CollisionMerge* end() const { return buffer + mergeSize; }

			void alloc(int x, int y)
			{
				mergeStride = y;
				mergeSize = x * y;
				buffer = new CollisionMerge[mergeSize];
			}

			void release() { if (buffer) delete[] buffer; }

			CollisionMerge* get(int x, int y) { return &buffer[x * mergeStride + y]; }

			void doMerge(SkinnedMeshShape* shape0, SkinnedMeshShape* shape1, CollisionResult* collisions, int count);
			void apply(SkinnedMeshBody* body0, SkinnedMeshBody* body1, CollisionDispatcher* dispatcher);

			int mergeStride;
			int mergeSize;
			CollisionMerge* buffer;
			std::mutex lock;
		};

		template<typename T, bool Swap>
		class CollisionRunner
		{
		public:
			CollisionRunner(
				PerVertexShape* shape0,
				T* shape1,
				std::shared_ptr<MergeBuffer> merge,
				std::shared_ptr<std::function<void()>> apply)
				: m_shape0(shape0), m_shape1(shape1), m_merge(merge), m_apply(apply)
			{
				ColliderTree* c0 = &shape0->m_tree;
				ColliderTree* c1 = &shape1->m_tree;

				std::vector<std::pair<ColliderTree*, ColliderTree*>> pairs;
				pairs.reserve(c0->colliders.size() + c1->colliders.size());
				c0->checkCollisionL(c1, pairs);
				if (pairs.empty()) return;
				int npairs = pairs.size();

				// Create buffers for collision processing
				m_collisionPair.reset(new CudaCollisionPair<T::CudaType>(
					shape0->m_cudaObject.get(),
					shape1->m_cudaObject.get(),
					npairs,
					&m_results));

				// Set up data for each pair of collision trees
				for (int i = 0; i < npairs; ++i)
				{
					auto a = pairs[i].first;
					auto b = pairs[i].second;
					auto asize = b->isKinematic ? a->dynCollider : a->numCollider;
					auto bsize = a->isKinematic ? b->dynCollider : b->numCollider;

					if (asize > 0 && bsize > 0)
					{
						m_collisionPair->addPair(
							pairs[i].first->cbuf - shape0->m_colliders.data(),
							pairs[i].second->cbuf - shape1->m_colliders.data(),
							asize,
							bsize,
							a->aabbMe,
							b->aabbMe);
					}
				}

				// Run the kernel
				m_collisionPair->launch();
			}

			void operator()()
			{
				int count = 0;
				if (m_collisionPair)
				{
					m_collisionPair->synchronize();
					for (int i = 0; count < MaxCollisionCount && i < m_collisionPair->numPairs(); ++i)
					{
						if (m_results[i].depth <= 0)
						{
							// Kernel doesn't know real collider addresses, so it sets colliders referenced to 0.
							// We need to convert them to corresponding addresses in the real host-side data.
							int ciA = m_results[i].colliderA - static_cast<Collider*>(0);
							int ciB = m_results[i].colliderB - static_cast<Collider*>(0);
							m_results[i].colliderA = m_shape0->m_colliders.data() + ciA;
							m_results[i].colliderB = m_shape1->m_colliders.data() + ciB;

							if (Swap)
							{
								std::swap(m_results[i].colliderA, m_results[i].colliderB);
								std::swap(m_results[i].posA, m_results[i].posB);
								m_results[i].normOnB = -m_results[i].normOnB;
							}
							std::swap(m_results[count++], m_results[i]);
						}
					}
				}

				std::lock_guard l(m_merge->lock);
				if (count > 0)
				{
					if (Swap)
					{
						m_merge->doMerge(m_shape1, m_shape0, m_results, count);
					}
					else
					{
						m_merge->doMerge(m_shape0, m_shape1, m_results, count);
					}
				}

				if (m_apply.use_count() == 1)
				{
					(*m_apply)();
				}
				m_apply = nullptr;
			}
		private:
			PerVertexShape* m_shape0;
			T* m_shape1;
			std::shared_ptr<MergeBuffer> m_merge;
			std::shared_ptr<std::function<void()>> m_apply;

			std::shared_ptr<CudaCollisionPair<typename T::CudaType>> m_collisionPair;
			CollisionResult* m_results;
		};

		template <class T0, class T1>
		static void processCollision(T0* shape0, T1* shape1, MergeBuffer& merge, CollisionResult* collision);
	};
}
