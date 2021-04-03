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
				cuCreateStream(&m_stream);
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

		template <typename CudaT, typename HostT = CudaT>
		class CudaBuffer
		{
		public:

			CudaBuffer(int n)
				: m_size(n * sizeof(CudaT))
			{
				static_assert(sizeof(CudaT) == sizeof(HostT), "Device and host types different sizes");
				cuGetDeviceBuffer(&reinterpret_cast<void*>(m_deviceData), m_size);
				cuGetHostBuffer(&reinterpret_cast<void*>(m_hostData), m_size);
			}

			~CudaBuffer()
			{
				cuFreeDevice(m_deviceData);
				cuFreeHost(m_hostData);
			}

			void toDevice(CudaStream& stream)
			{
				cuCopyToDevice(m_deviceData, m_hostData, m_size, stream);
			}

			void toHost(CudaStream& stream)
			{
				cuCopyToHost(m_hostData, m_deviceData, m_size, stream);
			}

			operator HostT* () { return m_hostData; }
			HostT* get() { return m_hostData; }

			CudaT* getD() { return m_deviceData; }

		private:

			int m_size;
			CudaT* m_deviceData;
			HostT* m_hostData;
		};
	}

	class CudaBody::Imp
	{
		friend class CudaPerTriangleShape::Imp;
		friend class CudaPerVertexShape::Imp;
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
			body->m_vpos.reset(m_vertexBuffer.get(), NullDeleter<VertexPos[]>());
		}

		void launch()
		{
			m_bones.toDevice(m_stream);
			cuRunBodyUpdate(
				m_stream,
				m_numVertices,
				m_vertexData.getD(),
				m_vertexBuffer.getD(),
				m_bones.getD());
			m_vertexBuffer.toHost(m_stream);
		}

	private:

		CudaStream m_stream;

		int m_numVertices;
		CudaBuffer<cuBone, Bone> m_bones;
		CudaBuffer<cuVertex, Vertex> m_vertexData;
		CudaBuffer<cuVector3, VertexPos> m_vertexBuffer;
	};

	CudaBody::CudaBody(SkinnedMeshBody* body)
		: m_imp(new Imp(body))
	{}

	void CudaBody::launch()
	{
		m_imp->launch();
	}

	class CudaPerTriangleShape::Imp
	{
	public:

		Imp(PerTriangleShape* shape)
			: m_numColliders(shape->m_colliders.size()),
			m_body(shape->m_owner->m_cudaObject->m_imp),
			m_input(shape->m_colliders.size()),
			m_output(shape->m_colliders.size())
		{
			for (int i = 0; i < m_numColliders; ++i)
			{
				m_input[i].vertexIndices[0] = shape->m_colliders[i].vertices[0];
				m_input[i].vertexIndices[1] = shape->m_colliders[i].vertices[1];
				m_input[i].vertexIndices[2] = shape->m_colliders[i].vertices[2];
				m_input[i].margin = shape->m_shapeProp.margin;
				m_input[i].penetration = shape->m_shapeProp.penetration;
			}
			m_input.toDevice(m_body->m_stream);

			Aabb* aabb = m_output.get();
			shape->m_tree.relocateAabb(aabb);
			shape->m_aabb.reset(aabb, NullDeleter<Aabb[]>());
		}

		void launch()
		{
			cuRunPerTriangleUpdate(
				m_body->m_stream,
				m_numColliders,
				m_input.getD(),
				m_output.getD(),
				m_body->m_vertexBuffer.getD());
			m_output.toHost(m_body->m_stream);
		}

	private:

		std::shared_ptr<CudaBody::Imp> m_body;

		int m_numColliders;
		CudaBuffer<cuPerTriangleInput> m_input;
		CudaBuffer<cuAabb, Aabb> m_output;
	};

	CudaPerTriangleShape::CudaPerTriangleShape(PerTriangleShape* shape)
		: m_imp(new Imp(shape))
	{}

	void CudaPerTriangleShape::launch()
	{
		m_imp->launch();
	}

	class CudaPerVertexShape::Imp
	{
	public:

		Imp(PerVertexShape* shape)
			: m_numColliders(shape->m_colliders.size()),
			m_body(shape->m_owner->m_cudaObject->m_imp),
			m_input(shape->m_colliders.size()),
			m_output(shape->m_colliders.size())
		{
			for (int i = 0; i < m_numColliders; ++i)
			{
				m_input[i].vertexIndex = shape->m_colliders[i].vertex;
				m_input[i].margin = shape->m_shapeProp.margin;
			}
			m_input.toDevice(m_body->m_stream);

			Aabb* aabb = m_output.get();
			shape->m_tree.relocateAabb(aabb);
			shape->m_aabb.reset(aabb, NullDeleter<Aabb[]>());
		}

		void launch()
		{
			cuRunPerVertexUpdate(
				m_body->m_stream,
				m_numColliders,
				m_input.getD(),
				m_output.getD(),
				m_body->m_vertexBuffer.getD());
			m_output.toHost(m_body->m_stream);
		}

	private:

		std::shared_ptr<CudaBody::Imp> m_body;

		int m_numColliders;
		CudaBuffer<cuPerVertexInput> m_input;
		CudaBuffer<cuAabb, Aabb> m_output;
	};

	CudaPerVertexShape::CudaPerVertexShape(PerVertexShape* shape)
		: m_imp(new Imp(shape))
	{}

	void CudaPerVertexShape::launch()
	{
		m_imp->launch();
	}

	CudaInterface* CudaInterface::instance()
	{
		static CudaInterface s_instance;
		return &s_instance;
	}

	bool CudaInterface::hasCuda()
	{
		return true;
	}

	void CudaInterface::synchronize()
	{
		cuSynchronize();
	}

	CudaInterface::CudaInterface()
	{}
}
