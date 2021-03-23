#pragma once

#include "hdtConvertNi.h"
#include "hdtSkinnedMesh/hdtSkinnedMeshBone.h"

namespace hdt
{
	class SkyrimBone : public SkinnedMeshBone
	{
	public:

		SkyrimBone(IDStr name, NiNode* node, btRigidBody::btRigidBodyConstructionInfo& ci);

		void resetTransformToOriginal() override;
		void readTransform(float timeStep) override;
		void writeTransform() override;

		int m_depth;
		Ref<NiNode> m_node;
	};
}
