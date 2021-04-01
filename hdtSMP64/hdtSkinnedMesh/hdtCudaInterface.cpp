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

		int numVertices = 0;
		cuPerVertexInput* perVertexInput = 0;
		cuPerVertexOutput* perVertexOutput = 0;

		void AllocatePerVertexData(int n)
		{
			if (n > numVertices)
			{
				if (perVertexInput)
				{
					cuFree(perVertexInput);
				}
				if (perVertexOutput)
				{
					cuFree(perVertexOutput);
				}
				numVertices = chunkSize * ((n - 1) / chunkSize + 1);
				cuGetBuffer(&perVertexInput, numVertices);
				cuGetBuffer(&perVertexOutput, numVertices);
			}
		}


		~CudaBuffers()
		{
			if (perVertexInput)
			{
				cuFree(perVertexInput);
			}
			if (perVertexOutput)
			{
				cuFree(perVertexOutput);
			}
		}
	};

	CudaInterface* CudaInterface::instance()
	{
		static CudaInterface s_instance;
		return &s_instance;
	}

	bool CudaInterface::hasCuda()
	{
		return true;
	}

	namespace
	{
		void setCuVector3(cuVector3& target, btVector3& source)
		{
			target.x = source.m_floats[0];
			target.y = source.m_floats[1];
			target.z = source.m_floats[2];
			target.w = source.m_floats[3];
		}

		void setM128(__m128& target, cuVector3& source)
		{
			target = _mm_setr_ps(source.x, source.y, source.z, source.w);
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

		m_buffers->AllocatePerVertexData(offsets.back());

		cuPerVertexInput* input = m_buffers->perVertexInput;
		cuPerVertexOutput* output = m_buffers->perVertexOutput;

		concurrency::parallel_for(0, n, [&](int i)
		{
			int j = offsets[i];
			float margin = vecShapes[i]->m_shapeProp.margin;
			auto& vertices = vecShapes[i]->m_owner->m_vpos;
			for (auto& c : vecShapes[i]->m_colliders)
			{
				setCuVector3(input[j].point, vertices[c.vertex].pos());
				input[j].margin = margin;
				++j;
			}
		});

		if (!cuRunPerVertexUpdate(offsets.back(), input, output))
		{
			_MESSAGE("CUDA kernel failed");
		}

		concurrency::parallel_for(0, n, [&](int i)
		{
			for (int j = 0; j < vecShapes[i]->m_colliders.size(); ++j)
			{
				setM128(vecShapes[i]->m_aabb[j].m_min, output[j + offsets[i]].aabbMin);
				setM128(vecShapes[i]->m_aabb[j].m_max, output[j + offsets[i]].aabbMax);
			}
			vecShapes[i]->m_tree.updateAabb();
		});
	}

	CudaInterface::CudaInterface()
		: m_buffers(new CudaBuffers())
	{}

}
