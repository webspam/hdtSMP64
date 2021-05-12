#include "hdtCudaInterface.h"

#include <ppl.h>
#include <immintrin.h>

struct cudaStream_t;

#include "hdtCudaCollision.cuh"

namespace hdt
{
	namespace
	{
		template<typename T>
		struct NullDeleter
		{
			void operator()(T*) const {}

			template<typename U>
			void operator()(U*) const {}
		};

		class CudaStream
		{
		public:
			CudaStream()
			{
				cuCreateStream(&m_stream).check(__FUNCTION__);
			}

			~CudaStream()
			{
				cuDestroyStream(m_stream);
			}

			void* get() { return m_stream; }
			operator void* () { return m_stream; }

		private:
			void* m_stream;
		};

		class CudaEvent
		{
		public:
			CudaEvent()
			{
				cuCreateEvent(&m_event).check(__FUNCTION__);
			}

			~CudaEvent()
			{
				cuDestroyEvent(m_event);
			}

			void record(CudaStream& stream)
			{
				cuRecordEvent(m_event, stream);
			}

			void wait()
			{
				cuWaitEvent(m_event);
			}

		private:
			void* m_event;
		};

		// CUDA buffer for long-lived objects
		template <typename CudaT, typename HostT = CudaT>
		class CudaBuffer
		{
		public:

			CudaBuffer(int n)
				: m_size(n * sizeof(CudaT))
			{
				static_assert(sizeof(CudaT) == sizeof(HostT), "Device and host types different sizes");
				cuGetDeviceBuffer(&reinterpret_cast<void*>(m_deviceData), m_size).check(__FUNCTION__);
				cuGetHostBuffer(&reinterpret_cast<void*>(m_hostData), m_size).check(__FUNCTION__);
			}

			~CudaBuffer()
			{
				cuFreeDevice(m_deviceData);
				cuFreeHost(m_hostData);
			}

			void toDevice(CudaStream& stream)
			{
				cuCopyToDevice(m_deviceData, m_hostData, m_size, stream).check(__FUNCTION__);
			}

			void toHost(CudaStream& stream)
			{
				cuCopyToHost(m_hostData, m_deviceData, m_size, stream).check(__FUNCTION__);
			}

			operator HostT* () { return m_hostData; }
			HostT* get() { return m_hostData; }

			CudaT* getD() { return m_deviceData; }

		private:

			int m_size;
			CudaT* m_deviceData;
			HostT* m_hostData;
		};

		template <typename CudaT>
		class CudaDeviceBuffer
		{
		public:

			CudaDeviceBuffer(int n)
				: m_size(n * sizeof(CudaT))
			{
				cuGetDeviceBuffer(&reinterpret_cast<void*>(m_deviceData), m_size);
			}

			~CudaDeviceBuffer()
			{
				cuFreeDevice(m_deviceData);
			}

			CudaT* getD() { return m_deviceData; }

		private:

			int m_size;
			CudaT* m_deviceData;
		};

		template <typename T, typename... Ts>
		class CudaDeviceBuffer<PlanarStruct<T, Ts...>>
		{
		public:

			CudaDeviceBuffer(int n)
				: m_size(32 * ((n - 1) / 32) + 1),
				m_buffer(m_size)
			{}

			PlanarStruct<T, Ts...> getD() {
				return PlanarStruct<T, Ts...>(m_buffer.getD(), m_size);
			}

		private:

			int m_size;
			CudaDeviceBuffer<T> m_buffer;
		};


		// Memory pool for small short-lived objects. This can grow arbitrarily in size, to the maximum required
		// in a single frame. All allocations get cleared at the end of the frame.
		class CudaBufferPool
		{
			using Buffers = std::pair<void*, void*>;
			using Record = std::tuple<size_t, const size_t, Buffers>;

			// Granularity for allocating blocks that won't fit a single page (however, this memory pool is
			// REALLY not designed for large allocations and using them is likely to leak memory badly)
			static constexpr size_t largeBlockSize = 1 << 20;

			// Page size for normal allocations
			static constexpr size_t pageSize = 1 << 24;

			// Granularity of small allocations, should match CUDA memory transaction size
			static constexpr size_t alignment = 128;

		public:

			static CudaBufferPool* instance()
			{
				static CudaBufferPool s_instance;
				return &s_instance;
			}

			std::pair<void*, void*> getBuffer(size_t size)
			{
				// FIXME: Locking for the whole method is lazy - should do something finer grained
				std::lock_guard l(m_lock);

				auto s = getSize(size);
				std::vector<Record>::iterator it;
				for (it = m_buffers.begin(); it != m_buffers.end(); ++it)
				{
					if (std::get<0>(*it) + s <= std::get<1>(*it))
					{
						break;
					}
				}
				if (it == m_buffers.end())
				{
					size_t newSize = std::max(pageSize, blockSize(size));
					m_buffers.push_back({ 0, newSize, {0,0} });
					cuGetDeviceBuffer(&(std::get<2>(m_buffers.back()).first), newSize).check(__FUNCTION__);
					cuGetHostBuffer(&(std::get<2>(m_buffers.back()).second), newSize).check(__FUNCTION__);
					it = m_buffers.end() - 1;
				}
				Buffers result = {
					static_cast<uint8_t*>(std::get<2>(*it).first) + std::get<0>(*it),
					static_cast<uint8_t*>(std::get<2>(*it).second) + std::get<0>(*it)
				};
				std::get<0>(*it) += s;
				return result;
			}

			void clear()
			{
				for (auto& record : m_buffers)
				{
					std::get<0>(record) = 0;
				}
			}

		private:

			constexpr size_t getSize(size_t size)
			{
				return alignment * ((size - 1) / alignment + 1);
			}

			constexpr size_t blockSize(size_t size)
			{
				return largeBlockSize * ((size - 1) / largeBlockSize + 1);
			}

			CudaBufferPool()
			{}

			~CudaBufferPool()
			{
				for (auto record : m_buffers)
				{
					cuFreeDevice(std::get<2>(record).first);
					cuFreeHost(std::get<2>(record).second);
				}
			}

			std::vector<Record> m_buffers;
			std::mutex m_lock;
		};

		// CUDA buffer for short-lived per-frame objects. There is no way to deallocate these explicitly - they
		// remain manually until the buffer pool is cleared at the end of the frame, and then all become unsafe.
		template <typename CudaT, typename HostT = CudaT>
		class CudaPooledBuffer
		{
		public:

			CudaPooledBuffer(size_t n)
				: m_size(n * sizeof(CudaT))
			{
				static_assert(sizeof(CudaT) == sizeof(HostT), "Device and host types different sizes");
				auto buffers = CudaBufferPool::instance()->getBuffer(m_size);
				m_deviceData = reinterpret_cast<CudaT*>(buffers.first);
				m_hostData = reinterpret_cast<HostT*>(buffers.second);
			}

			void toDevice(CudaStream& stream)
			{
				cuCopyToDevice(m_deviceData, m_hostData, m_size, stream).check(__FUNCTION__);
			}

			void toHost(CudaStream& stream)
			{
				cuCopyToHost(m_hostData, m_deviceData, m_size, stream).check(__FUNCTION__);
			}

			operator HostT* () { return m_hostData; }
			HostT* get() { return m_hostData; }

			CudaT* getD() { return m_deviceData; }

		private:

			size_t m_size;
			CudaT* m_deviceData;
			HostT* m_hostData;
		};
	}

	class CudaBody::Imp
	{
	public:

		Imp(SkinnedMeshBody* body)
			: m_numVertices(body->m_vertices.size()),
			m_bones(body->m_skinnedBones.size()),
			m_vertexData(body->m_vertices.size()),
			m_vertexBuffer(body->m_vertices.size())
		{
			std::copy(body->m_vertices.begin(), body->m_vertices.end(), m_vertexData.get());
			m_vertexData.toDevice(m_stream);

			body->m_bones.reset(m_bones.get(), NullDeleter<Bone[]>());
		}

		void launch()
		{
			m_bones.toDevice(m_stream);
			cuRunBodyUpdate(
				m_stream,
				m_numVertices,
				m_vertexData.getD(),
				m_vertexBuffer.getD(),
				m_bones.getD()).check(__FUNCTION__);
		}

		void synchronize()
		{
			cuSynchronize(m_stream).check(__FUNCTION__);
		}

		void recordState()
		{
			m_event.record(m_stream);
		}

		void waitForAabbData()
		{
			m_event.wait();
		}

		CudaStream m_stream;
		CudaDeviceBuffer<PlanarVectorArray> m_vertexBuffer;

	private:

		CudaEvent m_event;

		int m_numVertices;
		CudaBuffer<cuBone, Bone> m_bones;
		CudaBuffer<cuVertex, Vertex> m_vertexData;
	};

	CudaBody::CudaBody(SkinnedMeshBody* body)
		: m_imp(new Imp(body))
	{}

	void CudaBody::launch()
	{
		m_imp->launch();
	}

	void CudaBody::synchronize()
	{
		m_imp->synchronize();
	}

	void CudaBody::waitForAaabData()
	{
		m_imp->waitForAabbData();
	}

	void CudaBody::recordState()
	{
		m_imp->recordState();
	}

	class CudaColliderTree
	{
		using NodePair = std::pair<int, int>;

		ColliderTree* m_tree;
		int m_numNodes;
		int m_largestNode;

	public:

		CudaColliderTree(ColliderTree* tree, CudaStream& stream)
			: m_tree(tree),
			m_numNodes(nodeCount(*tree)),
			m_nodeData(m_numNodes),
			m_nodeAabbs(m_numNodes)
		{
			std::vector<NodePair> nodeData;
			buildNodeData(*tree, m_nodeData.get());
			m_nodeData.toDevice(stream);
		}

		void launch(CudaStream& stream, PlanarBoundingBoxArray boundingBoxes)
		{
			cuRunBoundingBoxReduce(stream, m_numNodes, m_nodeData.getD(), boundingBoxes, m_nodeAabbs.getD()).check(__FUNCTION__);
			m_nodeAabbs.toHost(stream);
		}

		void update()
		{
			updateBoundingBoxes(*m_tree, m_nodeAabbs);
		}

		CudaBuffer<NodePair> m_nodeData;
		CudaBuffer<cuAabb, Aabb> m_nodeAabbs;

	private:

		static int nodeCount(ColliderTree& tree)
		{
			int count = tree.numCollider ? 1 : 0;
			for (auto& child : tree.children)
			{
				count += nodeCount(child);
			}
			return count;
		}

		NodePair* buildNodeData(ColliderTree& tree, NodePair* nodeData)
		{
			if (tree.numCollider)
			{
				*nodeData++ = { tree.aabb - m_tree->aabb, tree.numCollider };
			}
			for (auto& child : tree.children)
			{
				nodeData = buildNodeData(child, nodeData);
			}
			return nodeData;
		}

		Aabb* updateBoundingBoxes(ColliderTree& tree, Aabb* boundingBoxes)
		{
			if (tree.numCollider)
			{
				tree.aabbMe = *boundingBoxes++;
			}
			else
			{
				tree.aabbMe.invalidate();
			}
			tree.aabbAll = tree.aabbMe;
			for (auto& child : tree.children)
			{
				boundingBoxes = updateBoundingBoxes(child, boundingBoxes);
				tree.aabbAll.merge(child.aabbAll);
			}
			return boundingBoxes;
		}
	};

	class CudaPerTriangleShape::Imp
	{
	public:

		Imp(PerTriangleShape* shape)
			: m_numColliders(shape->m_colliders.size()),
			m_penetrationType(shape->m_shapeProp.penetration > FLT_EPSILON ? eInternal : shape->m_shapeProp.penetration < -FLT_EPSILON ? eExternal : eNone),
			m_body(shape->m_owner->m_cudaObject->m_imp),
			m_input(shape->m_colliders.size()),
			m_output(shape->m_colliders.size()),
			m_tree(&shape->m_tree, m_body->m_stream)
		{
			for (int i = 0; i < m_numColliders; ++i)
			{
				m_input[i].vertexIndices[0] = shape->m_colliders[i].vertices[0];
				m_input[i].vertexIndices[1] = shape->m_colliders[i].vertices[1];
				m_input[i].vertexIndices[2] = shape->m_colliders[i].vertices[2];
				m_input[i].margin = shape->m_shapeProp.margin;
				m_input[i].penetration = -abs(shape->m_shapeProp.penetration);
			}
			m_input.toDevice(m_body->m_stream);
			m_tree.m_nodeData.toDevice(m_body->m_stream);
		}

		void launch()
		{
			cuRunPerTriangleUpdate(
				m_body->m_stream,
				m_numColliders,
				m_input.getD(),
				m_output.getD(),
				m_body->m_vertexBuffer.getD()).check(__FUNCTION__);
		}

		void launchTree()
		{
			m_tree.launch(m_body->m_stream, m_output.getD());
		}

		void updateTree()
		{
			m_tree.update();
		}

		CudaBuffer<cuPerTriangleInput> m_input;
		CudaDeviceBuffer<PlanarBoundingBoxArray> m_output;
		std::shared_ptr<CudaBody::Imp> m_body;
		const cuPenetrationType m_penetrationType;
	private:

		int m_numColliders;
		CudaColliderTree m_tree;

	};

	CudaPerTriangleShape::CudaPerTriangleShape(PerTriangleShape* shape)
		: m_imp(new Imp(shape))
	{}

	void CudaPerTriangleShape::launch()
	{
		m_imp->launch();
	}

	void CudaPerTriangleShape::launchTree()
	{
		m_imp->launchTree();
	}

	void CudaPerTriangleShape::updateTree()
	{
		m_imp->updateTree();
	}

	class CudaPerVertexShape::Imp
	{
	public:

		Imp(PerVertexShape* shape)
			: m_numColliders(shape->m_colliders.size()),
			m_body(shape->m_owner->m_cudaObject->m_imp),
			m_input(shape->m_colliders.size()),
			m_output(shape->m_colliders.size()),
			m_tree(&shape->m_tree, m_body->m_stream)
		{
			for (int i = 0; i < m_numColliders; ++i)
			{
				m_input[i].vertexIndex = shape->m_colliders[i].vertex;
				m_input[i].margin = shape->m_shapeProp.margin;
			}
			m_input.toDevice(m_body->m_stream);
			m_tree.m_nodeData.toDevice(m_body->m_stream);
		}

		void launch()
		{
			cuRunPerVertexUpdate(
				m_body->m_stream,
				m_numColliders,
				m_input.getD(),
				m_output.getD(),
				m_body->m_vertexBuffer.getD()).check(__FUNCTION__);
		}

		void launchTree()
		{
			m_tree.launch(m_body->m_stream, m_output.getD());
		}

		void updateTree()
		{
			m_tree.update();
		}

		CudaBuffer<cuPerVertexInput> m_input;
		CudaDeviceBuffer<PlanarBoundingBoxArray> m_output;
		std::shared_ptr<CudaBody::Imp> m_body;
		int m_numColliders;

	private:

		CudaColliderTree m_tree;
	};

	CudaPerVertexShape::CudaPerVertexShape(PerVertexShape* shape)
		: m_imp(new Imp(shape))
	{}

	void CudaPerVertexShape::launch()
	{
		m_imp->launch();
	}

	void CudaPerVertexShape::launchTree()
	{
		m_imp->launchTree();
	}

	void CudaPerVertexShape::updateTree()
	{
		m_imp->updateTree();
	}

	template <typename T>
	class CudaCollisionPair<T>::Imp
	{
	public:

		Imp(
			CudaPerVertexShape* shapeA,
			T* shapeB,
			int numCollisionPairs,
			CollisionResult** results)
			: m_shapeA(shapeA),
			m_shapeB(shapeB),
			m_numCollisionPairs(numCollisionPairs),
			m_nextPair(0),
			m_stream(),
			m_resultBuffer(numCollisionPairs),
			m_setupBuffer(numCollisionPairs)
		{
			*results = m_resultBuffer.get();
		}

		void addPair(
			int offsetA,
			int offsetB,
			int sizeA,
			int sizeB,
			const Aabb& aabbA,
			const Aabb& aabbB)
		{
			static_assert(sizeof(cuCollider) == sizeof(Collider));

			m_setupBuffer[m_nextPair++] = {
				sizeA,
				sizeB,
				offsetA,
				offsetB,
				*reinterpret_cast<const cuAabb*>(&aabbA),
				*reinterpret_cast<const cuAabb*>(&aabbB)
			};
		}

		void launch()
		{
			if (m_nextPair > 0)
			{
				m_setupBuffer.toDevice(m_stream);

				collisionFunc()(
					m_stream,
					m_nextPair,
					m_setupBuffer.getD(),
					m_shapeA->m_imp->m_input.getD(),
					m_shapeB->m_imp->m_input.getD(),
					m_shapeA->m_imp->m_output.getD(),
					m_shapeB->m_imp->m_output.getD(),
					m_shapeA->m_imp->m_body->m_vertexBuffer.getD(),
					m_shapeB->m_imp->m_body->m_vertexBuffer.getD(),
					m_resultBuffer.getD()).check(__FUNCTION__);

				m_resultBuffer.toHost(m_stream);
			}
		}

		void synchronize()
		{
			cuSynchronize(m_stream).check(__FUNCTION__);
		}

		int numPairs()
		{
			return m_nextPair;
		}

	private:

		CudaPerVertexShape* m_shapeA;
		T* m_shapeB;
		int m_numCollisionPairs;
		int m_nextPair;

		CudaStream m_stream;
		CudaPooledBuffer<cuCollisionResult, CollisionResult> m_resultBuffer;
		CudaPooledBuffer<cuCollisionSetup> m_setupBuffer;

		auto collisionFunc() -> decltype(cuRunCollision<eNone, std::remove_pointer<decltype(m_shapeB->m_imp->m_input.getD())>::type>)*;
	};

	template<>
	auto CudaCollisionPair<CudaPerVertexShape>::Imp::collisionFunc()
		-> decltype(cuRunCollision<eNone, cuPerVertexInput>)*
	{
		return cuRunCollision<eNone, cuPerVertexInput>;
	}

	template<>
	auto CudaCollisionPair<CudaPerTriangleShape>::Imp::collisionFunc()
		-> decltype(cuRunCollision<eNone, cuPerTriangleInput>)*
	{
		switch (m_shapeB->m_imp->m_penetrationType)
		{
		case eNone:
			return cuRunCollision<eNone, cuPerTriangleInput>;
		case eInternal:
			return cuRunCollision<eInternal, cuPerTriangleInput>;
		case eExternal:
		default:
			return cuRunCollision<eExternal, cuPerTriangleInput>;
		}
	}

	template <typename T>
	CudaCollisionPair<T>::CudaCollisionPair(
		CudaPerVertexShape* shapeA,
		T* shapeB, 
		int numCollisionPairs,
		CollisionResult** results)
		: m_imp(new Imp(shapeA, shapeB, numCollisionPairs, results))
	{}

	template <typename T>
	void CudaCollisionPair<T>::addPair(
		int offsetA,
		int offsetB,
		int sizeA,
		int sizeB,
		const Aabb& aabbA,
		const Aabb& aabbB)
	{
		m_imp->addPair(offsetA, offsetB, sizeA, sizeB, aabbA, aabbB);
	}

	template <typename T>
	void CudaCollisionPair<T>::launch()
	{
		m_imp->launch();
	}

	template <typename T>
	void CudaCollisionPair<T>::synchronize()
	{
		m_imp->synchronize();
	}

	template <typename T>
	int CudaCollisionPair<T>::numPairs()
	{
		return m_imp->numPairs();
	}

	bool CudaInterface::enableCuda = false;

	CudaInterface* CudaInterface::instance()
	{
		static CudaInterface s_instance;
		return &s_instance;
	}

	bool CudaInterface::hasCuda()
	{
		return enableCuda && m_enabled;
	}

	void CudaInterface::synchronize()
	{
		cuSynchronize().check(__FUNCTION__);
	}

	void CudaInterface::clearBufferPool()
	{
		CudaBufferPool::instance()->clear();
	}

	CudaInterface::CudaInterface()
		: m_enabled(cuDeviceCount() > 0)
	{
		if (m_enabled)
		{
			cuInitialize();
		}
	}

	template class CudaCollisionPair<CudaPerVertexShape>;
	template class CudaCollisionPair<CudaPerTriangleShape>;
}
