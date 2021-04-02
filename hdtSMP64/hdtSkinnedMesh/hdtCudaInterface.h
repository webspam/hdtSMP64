#pragma once

#include "hdtSkinnedMeshShape.h"

namespace hdt
{
	class CudaBody
	{
		friend class CudaPerTriangleShape;
		friend class CudaPerVertexShape;
	public:
		CudaBody(SkinnedMeshBody* body);

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};
	
	class CudaPerTriangleShape
	{
		friend class CudaBody;
	public:
		CudaPerTriangleShape(PerTriangleShape* shape);

		void launch();

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};

	class CudaPerVertexShape
	{
		friend class CudaBody;
	public:
		CudaPerVertexShape(PerVertexShape* shape);

		void launch();

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};

	class CudaInterface
	{
		struct CudaBuffers;

	public:

		static CudaInterface* instance();

		bool hasCuda();

		void synchronize();

	private:

		CudaInterface();
	};
}
