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
		void launch();

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};
	
	class CudaPerTriangleShape
	{
	public:
		class Imp;

		CudaPerTriangleShape(PerTriangleShape* shape);
		void launch();

	private:
		std::shared_ptr<Imp> m_imp;
	};

	class CudaPerVertexShape
	{
	public:
		class Imp;

		CudaPerVertexShape(PerVertexShape* shape);
		void launch();

	private:
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
