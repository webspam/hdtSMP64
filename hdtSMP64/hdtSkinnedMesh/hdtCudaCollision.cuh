#pragma once

#include <memory>
#include <immintrin.h>

namespace hdt
{
	union cuVector3
	{
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		__m128 val;
	};

	struct cuTriangle
	{
		cuVector3 pA;
		cuVector3 pB;
		cuVector3 pC;
	};

	struct cuPerVertexInput
	{
		int vertexIndex;
		float margin;
	};

	struct cuPerTriangleInput
	{
		int vertexIndices[3];
		float margin;
		float penetration;
	};

	struct cuAabb
	{
		cuVector3 aabbMin;
		cuVector3 aabbMax;
	};

	void cuCreateStream(void** ptr);

	void cuDestroyStream(void* ptr);

	template<typename T>
	void cuGetBuffer(T** buf, int size);
	
	template<typename T>
	void cuFree(T* buf);

	bool cuRunPerVertexUpdate(int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData);

	bool cuRunPerTriangleUpdate(int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData);

	bool cuSynchronize();
}
