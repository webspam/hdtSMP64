#include "hdtCudaInterface.h"
#include "hdtCudaCollision.cuh"

#include <ppl.h>
#include <immintrin.h>

namespace hdt
{
	struct CudaInterface::CudaBuffers
	{
		// Allocate buffers in chunks of this size, so we don't keep reallocating them too much
		static const int chunkSize = 1 << 16;

		template<typename T>
		struct Buffer
		{
			int size = 0;
			T* data = 0;

			void Allocate(int n)
			{
				if (n > size)
				{
					if (data)
					{
						cuFree(data);
					}
					size = chunkSize * ((n - 1) / chunkSize + 1);
					cuGetBuffer(&data, size);
				}
			}

			operator T* () { return data; }

			~Buffer()
			{
				if (data)
				{
					cuFree(data);
				}
			}
		};

		Buffer<cuVector3> vertexData;

		Buffer<cuPerVertexInput> perVertexInput;
		Buffer<cuAabb> perVertexOutput;

		Buffer<cuPerTriangleInput> perTriangleInput;
		Buffer<cuAabb> perTriangleOutput;
	};

	CudaInterface* CudaInterface::instance()
	{
		static CudaInterface s_instance;
		return &s_instance;
	}

	bool CudaInterface::hasCuda()
	{
		return false;
	}

	namespace
	{
		template<typename T>
		struct NullDeleter
		{
			void operator()(T*) const {}

			template<typename U>
			void operator()(U*) const {}
		};
	}

	void CudaInterface::assignVertexSpace(std::unordered_set<SkinnedMeshBody*> bodies)
	{
		int count = 0;
		for (auto body : bodies)
		{
			count += body->m_vertices.size();
		}
		m_buffers->vertexData.Allocate(count);

		int offset = 0;
		for (auto body : bodies)
		{
			body->m_vpos.reset(reinterpret_cast<VertexPos*>(m_buffers->vertexData + offset), NullDeleter<VertexPos[]>());
			offset += body->m_vertices.size();
		}
	}

	void CudaInterface::perVertexUpdate(std::unordered_set<PerVertexShape*> shapes)
	{
		std::vector<PerVertexShape*> vecShapes(shapes.begin(), shapes.end());
		std::vector<int> offsets;
		int n = static_cast<int>(vecShapes.size());

		offsets.reserve(n + 1);
		offsets.push_back(0);
		for (auto shape : vecShapes)
		{
			offsets.push_back(offsets.back() + shape->m_colliders.size());
		}

		m_buffers->perVertexInput.Allocate(offsets.back());
		m_buffers->perVertexOutput.Allocate(offsets.back());

		cuPerVertexInput* input = m_buffers->perVertexInput;
		cuAabb* output = m_buffers->perVertexOutput;
		cuVector3* vertexData = m_buffers->vertexData;

		concurrency::parallel_for(0, n, [&](int i)
		{
			int firstVertex = reinterpret_cast<cuVector3*>(vecShapes[i]->m_owner->m_vpos.get()) - m_buffers->vertexData;

			int j = offsets[i];
			float margin = vecShapes[i]->m_shapeProp.margin;
			for (auto& c : vecShapes[i]->m_colliders)
			{
				input[j].vertexIndex = firstVertex + c.vertex;
				input[j].margin = margin;
				++j;
			}
		});

		if (!cuRunPerVertexUpdate(offsets.back(), input, output, vertexData))
		{
			_MESSAGE("CUDA kernel failed");
		}

		concurrency::parallel_for(0, n, [&](int i)
		{
			for (int j = 0; j < vecShapes[i]->m_colliders.size(); ++j)
			{
				vecShapes[i]->m_aabb[j].m_min = output[j + offsets[i]].aabbMin.val;
				vecShapes[i]->m_aabb[j].m_max = output[j + offsets[i]].aabbMax.val;
			}
			vecShapes[i]->m_tree.updateAabb();
		});
	}

	void CudaInterface::perTriangleUpdate(std::unordered_set<PerTriangleShape*> shapes)
	{
		std::vector<PerTriangleShape*> vecShapes(shapes.begin(), shapes.end());
		std::vector<int> offsets;
		int n = static_cast<int>(vecShapes.size());

		offsets.reserve(n + 1);
		offsets.push_back(0);
		for (auto shape : vecShapes)
		{
			offsets.push_back(offsets.back() + shape->m_colliders.size());
		}

		m_buffers->perTriangleInput.Allocate(offsets.back());
		m_buffers->perTriangleOutput.Allocate(offsets.back());

		cuPerTriangleInput* input = m_buffers->perTriangleInput;
		cuAabb* output = m_buffers->perVertexOutput;
		cuVector3* vertexData = m_buffers->vertexData;

		concurrency::parallel_for(0, n, [&](int i)
		{
			int firstVertex = reinterpret_cast<cuVector3*>(vecShapes[i]->m_owner->m_vpos.get()) - m_buffers->vertexData;

			int j = offsets[i];
			float margin = vecShapes[i]->m_shapeProp.margin;
			float penetration = vecShapes[i]->m_shapeProp.penetration;
			for (auto& c : vecShapes[i]->m_colliders)
			{
				input[j].vertexIndices[0] = firstVertex + c.vertices[0];
				input[j].vertexIndices[1] = firstVertex + c.vertices[1];
				input[j].vertexIndices[2] = firstVertex + c.vertices[2];
				input[j].margin = margin;
				input[j].penetration = penetration;
				++j;
			}
		});

		if (!cuRunPerTriangleUpdate(offsets.back(), input, output, vertexData))
		{
			_MESSAGE("CUDA kernel failed");
		}

		concurrency::parallel_for(0, n, [&](int i)
		{
			for (int j = 0; j < vecShapes[i]->m_colliders.size(); ++j)
			{
				vecShapes[i]->m_aabb[j].m_min = output[j + offsets[i]].aabbMin.val;
				vecShapes[i]->m_aabb[j].m_max = output[j + offsets[i]].aabbMax.val;
			}
			vecShapes[i]->m_tree.updateAabb();
		});
	}

	CudaInterface::CudaInterface()
		: m_buffers(new CudaBuffers())
	{}

}
