#pragma once

#include "hdtBulletHelper.h"

namespace hdt
{
	_CRT_ALIGN(16) struct Bone
	{
		// cache from rigidbody
		btMatrix4x3T m_vertexToWorld;
	};
}
