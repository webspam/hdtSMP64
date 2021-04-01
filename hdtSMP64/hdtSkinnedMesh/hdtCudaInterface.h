#pragma once

#include "hdtSkinnedMeshShape.h"

namespace hdt
{
	class CudaInterface
	{
		struct CudaBuffers;

	public:

		static CudaInterface* instance();

		bool hasCuda();

		void assignVertexSpace(std::unordered_set<SkinnedMeshBody*> bodies);

		void perVertexUpdate(std::unordered_set<PerVertexShape*> shapes);

		void perTriangleUpdate(std::unordered_set<PerTriangleShape*> shapes);

	private:

		CudaInterface();

		std::unique_ptr<CudaBuffers> m_buffers;
	};
}
