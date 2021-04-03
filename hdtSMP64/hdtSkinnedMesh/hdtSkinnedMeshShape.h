#pragma once

#include "hdtCollisionAlgorithm.h"
#include "hdtCollider.h"
#include "hdtSkinnedMeshBody.h"

namespace hdt
{
	class PerVertexShape;
	class PerTriangleShape;
	class CudaPerVertexShape;
	class CudaPerTriangleShape;

	class SkinnedMeshShape : public RefObject
	{
	public:
		BT_DECLARE_ALIGNED_ALLOCATOR();

		SkinnedMeshShape(SkinnedMeshBody* body);
		virtual ~SkinnedMeshShape();

		virtual PerVertexShape* asPerVertexShape() { return nullptr; }
		virtual PerTriangleShape* asPerTriangleShape() { return nullptr; }

		const Aabb& getAabb() const { return m_tree.aabbAll; }

		virtual void clipColliders();
		virtual void finishBuild() = 0;
		virtual void internalUpdate() = 0;
		virtual int getBonePerCollider() = 0;
		virtual void markUsedVertices(bool* flags) = 0;
		virtual void remapVertices(UINT* map) = 0;

		virtual float getColliderBoneWeight(const Collider* c, int boneIdx) = 0;
		virtual int getColliderBoneIndex(const Collider* c, int boneIdx) = 0;

		SkinnedMeshBody* m_owner;
		std::shared_ptr<Aabb[]> m_aabb;
		vectorA16<Collider> m_colliders;
		ColliderTree m_tree;
		float m_windEffect = 0.f;

#ifdef ENABLE_CL
		cl::Buffer		m_aabbCL;
		cl::Buffer		m_colliderCL;
		cl::Event		m_eDoneCL;
		virtual void internalUpdateCL() = 0;
#endif
	};

	class PerVertexShape : public SkinnedMeshShape
	{
	public:
		PerVertexShape(SkinnedMeshBody* body);
		virtual ~PerVertexShape();

		PerVertexShape* asPerVertexShape() override { return this; }

		void internalUpdate() override;
		int getBonePerCollider() override { return 4; }

		float getColliderBoneWeight(const Collider* c, int boneIdx) override
		{
			return m_owner->m_vertices[c->vertex].m_weight[boneIdx];
		}

		int getColliderBoneIndex(const Collider* c, int boneIdx) override
		{
			return m_owner->m_vertices[c->vertex].getBoneIdx(boneIdx);
		}

		void finishBuild() override;
		void markUsedVertices(bool* flags) override;
		void remapVertices(UINT* map) override;

		void autoGen();

		struct ShapeProp
		{
			float margin = 1.0f;
		} m_shapeProp;

		std::shared_ptr<CudaPerVertexShape> m_cudaObject;

#ifdef ENABLE_CL
		static hdtCLKernel		m_kernel;
		virtual void internalUpdateCL();
#endif
	};


	class PerTriangleShape : public SkinnedMeshShape
	{
	public:
		PerTriangleShape(SkinnedMeshBody* body);
		virtual ~PerTriangleShape();

		PerVertexShape* asPerVertexShape() override { return m_verticesCollision; }
		PerTriangleShape* asPerTriangleShape() override { return this; }

		void internalUpdate() override;
		int getBonePerCollider() override { return 12; }

		float getColliderBoneWeight(const Collider* c, int boneIdx) override
		{
			return m_owner->m_vertices[c->vertices[boneIdx / 4]].m_weight[boneIdx % 4];
		}

		int getColliderBoneIndex(const Collider* c, int boneIdx) override
		{
			return m_owner->m_vertices[c->vertices[boneIdx / 4]].getBoneIdx(boneIdx % 4);
		}

		void finishBuild() override;
		void markUsedVertices(bool* flags) override;
		void remapVertices(UINT* map) override;

		void addTriangle(int p0, int p1, int p2);

		struct ShapeProp
		{
			float margin = 1.0f;
			float penetration = 1.f;
		} m_shapeProp;

		Ref<PerVertexShape> m_verticesCollision;

		std::shared_ptr<CudaPerTriangleShape> m_cudaObject;

#ifdef ENABLE_CL
		static hdtCLKernel		m_kernel;
		virtual void internalUpdateCL();
#endif
	};
}
