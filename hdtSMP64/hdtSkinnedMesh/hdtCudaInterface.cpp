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

		template <typename T>
		class CudaBuffer
		{
		public:

			CudaBuffer(int n)
			{
				cuGetBuffer(&m_data, n);
			}

			~CudaBuffer()
			{
				cuFree(m_data);
			}

			operator T* () { return m_data; }
			T* get() { return m_data; }

		private:

			T* m_data;
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

		private:
			void* m_stream;
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
			std::copy(body->m_vertices.begin(), body->m_vertices.end(), reinterpret_cast<Vertex*>(m_vertexData.get()));
			body->m_bones.reset(reinterpret_cast<Bone*>(m_bones.get()), NullDeleter<Bone[]>());
			body->m_vpos.reset(reinterpret_cast<VertexPos*>(m_vertexBuffer.get()), NullDeleter<VertexPos[]>());
		}

		void launch()
		{
			cuRunBodyUpdate(
				m_stream.get(),
				m_numVertices,
				m_vertexData,
				m_vertexBuffer,
				m_bones);
		}

	private:

		CudaStream m_stream;

		int m_numVertices;
		CudaBuffer<cuBone> m_bones;
		CudaBuffer<cuVertex> m_vertexData;
		CudaBuffer<cuVector3> m_vertexBuffer;
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

			Aabb* aabb = reinterpret_cast<Aabb*>(m_output.get());
			shape->m_tree.relocateAabb(aabb);
			shape->m_aabb.reset(aabb, NullDeleter<Aabb[]>());
		}

		void launch()
		{
			cuRunPerTriangleUpdate(
				m_body->m_stream.get(),
				m_numColliders,
				m_input,
				m_output,
				m_body->m_vertexBuffer);
		}

	private:

		std::shared_ptr<CudaBody::Imp> m_body;

		int m_numColliders;
		CudaBuffer<cuPerTriangleInput> m_input;
		CudaBuffer<cuAabb> m_output;
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

			Aabb* aabb = reinterpret_cast<Aabb*>(m_output.get());
			shape->m_tree.relocateAabb(aabb);
			shape->m_aabb.reset(aabb, NullDeleter<Aabb[]>());
		}

		void launch()
		{
			cuRunPerVertexUpdate(
				m_body->m_stream.get(),
				m_numColliders,
				m_input,
				m_output,
				m_body->m_vertexBuffer);
		}

	private:

		std::shared_ptr<CudaBody::Imp> m_body;

		int m_numColliders;
		CudaBuffer<cuPerVertexInput> m_input;
		CudaBuffer<cuAabb> m_output;
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
		return false;
	}

	void CudaInterface::synchronize()
	{
		cuSynchronize();
	}

	CudaInterface::CudaInterface()
	{}
}
