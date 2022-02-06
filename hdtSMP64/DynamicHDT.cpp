#include "DynamicHDT.h"
#include "hdtSkyrimSystem.h"
#include "hdtSkinnedMesh/hdtSkinnedMeshSystem.h"

UInt32 hdt::util::splitArmorAddonFormID(std::string nodeName)
{
	try {
		return std::stoul(nodeName.substr(1, 8), nullptr, 16);
	}
	catch (...) {
		return 0;
	}
}

std::string hdt::util::UInt32toString(UInt32 formID)
{
	char buffer[16];
	sprintf_s(buffer, "%08X", formID);
	return std::string(buffer);
}

void hdt::util::transferCurrentPosesBetweenSystems(hdt::SkyrimSystem* src, hdt::SkyrimSystem* dst)
{
	for (auto& b1 : src->getBones()) {
		for (auto& b2 : dst->getBones()) {
			if (b1->m_name && b1->m_name == b2->m_name) {
				if (b2->m_rig.isStaticOrKinematicObject())break;
				if (b1->m_rig.isStaticOrKinematicObject())break;

				b2->m_rig.setWorldTransform(b1->m_rig.getWorldTransform());
				b2->m_rig.setAngularVelocity(b1->m_rig.getAngularVelocity());
				b2->m_rig.setLinearVelocity(b1->m_rig.getLinearVelocity());
				b2->m_localToRig = b1->m_localToRig;
				b2->m_rigToLocal = b1->m_rigToLocal;
				break;
			}
		}
	}
}
