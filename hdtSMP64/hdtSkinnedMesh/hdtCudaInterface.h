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

		void perVertexUpdate(std::unordered_set<PerVertexShape*> shapes);

	private:

		CudaInterface();

		std::unique_ptr<CudaBuffers> m_buffers;
	};
}
