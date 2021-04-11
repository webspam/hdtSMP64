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
		void synchronize();
		void waitForAaabData();
		void launchTransfer();

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};
	
	class CudaPerTriangleShape
	{
		template <typename T>
		friend class CudaCollisionPair;
	public:
		class Imp;

		CudaPerTriangleShape(PerTriangleShape* shape);
		void launch();

	private:
		std::shared_ptr<Imp> m_imp;
	};

	class CudaPerVertexShape
	{
		template <typename T>
		friend class CudaCollisionPair;
	public:
		class Imp;

		CudaPerVertexShape(PerVertexShape* shape);
		void launch();

	private:
		std::shared_ptr<Imp> m_imp;
	};

	template <typename T>
	class CudaCollisionPair
	{
	public:
		CudaCollisionPair(
			CudaPerVertexShape* shapeA,
			T* shapeB,
			int numCollisionPairs,
			int numColliders,
			CollisionResult** results,
			int** indexData);

		void addPair(
			int offsetA,
			int offsetB,
			int sizeA,
			int sizeB);

		void sendVertexLists();

		void launch();

		void synchronize();

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

		void clearBufferPool();

	private:

		CudaInterface();
	};
}
