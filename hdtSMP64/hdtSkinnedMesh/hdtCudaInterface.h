#pragma once

#include "hdtSkinnedMeshShape.h"

namespace hdt
{
	struct CudaCollisionData
	{
		int offset;
		int size;
		Aabb boundingBox;
	};

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
		void launchTransfer();
		void updateTree();

	private:
		std::shared_ptr<Imp> m_imp;
	};

	class CudaPerVertexShape
	{
		template <typename T>
		friend class CudaCollisionPair;
		template <typename T>
		friend class CudaCollisionPair2;
	public:
		class Imp;

		CudaPerVertexShape(PerVertexShape* shape);
		void launch();
		void launchTransfer();
		void updateTree();

	private:
		std::shared_ptr<Imp> m_imp;

	public:
		CudaCollisionData* const m_collisionData;

	};

	template <typename T>
	class CudaCollisionPair2
	{
	public:
		CudaCollisionPair2(
			CudaPerVertexShape* shapeA,
			T* shapeB,
			int numCollisionPairs,
			CollisionResult** results);

		void sendColliderGroups(int endKinematic, int startdynamic);
		void launchBoundingBoxCheck(Aabb& boundingBox);
		void launchCollision(int offset, int numDynamic, int numColliders);

		void synchronize();

	private:
		class Imp;
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
