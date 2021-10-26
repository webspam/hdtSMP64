#pragma once

#include "hdtCollider.h"

namespace hdt
{
	struct CollisionResult
	{
		btVector3 posA;
		btVector3 posB;
		btVector3 normOnB;
		Collider* colliderA;
		Collider* colliderB;
		float depth;
	};

	struct CheckTriangle
	{
		CheckTriangle(const btVector3& p0, const btVector3& p1, const btVector3& p2, float margin, float prenetration);

		btVector3 p0, p1, p2, normal;
		float margin, prenetration;
		bool valid;
	};

	bool checkSphereSphere(const btVector3& a, const btVector3& b, float ra, float rb, CollisionResult& res);
	bool checkSphereTriangle(const btVector3& s, float r, const CheckTriangle& tri, CollisionResult& res);
}
