#pragma once

#include "hdtSkinnedMeshShape.h"
#include "hdtDispatcher.h"

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
		void recordState();

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
		void launchTree();
		void updateTree();

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
		void launchTree();
		void updateTree();

	private:
		std::shared_ptr<Imp> m_imp;
	};

	class CudaMergeBuffer
	{
		template <typename T>
		friend class CudaCollisionPair;
	public:
		class Imp;

		CudaMergeBuffer(int x, int y);

		void launchTransfer();

		void apply(SkinnedMeshBody* body0, SkinnedMeshBody* body1, CollisionDispatcher* dispatcher);

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
			int numCollisionPairs);

		void addPair(
			int offsetA,
			int offsetB,
			int sizeA,
			int sizeB,
			const Aabb& aabbA,
			const Aabb& aabbB);

		void launch(CudaMergeBuffer* merge, bool swap);

		int numPairs();

	private:
		class Imp;
		std::shared_ptr<Imp> m_imp;
	};

	class CudaInterface
	{
		struct CudaBuffers;

	public:
		static bool enableCuda;

		static CudaInterface* instance();

		bool hasCuda();

		void synchronize();

		void clearBufferPool();

	private:

		CudaInterface();
		bool m_enabled;
	};
}
