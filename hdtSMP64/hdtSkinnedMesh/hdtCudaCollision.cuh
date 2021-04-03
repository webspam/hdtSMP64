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

	struct cuBone
	{
		cuVector3 transform[4];
		cuVector3 marginMultiplier; // Note only w component actually used
	};

	struct cuVertex
	{
		cuVector3 position;
		float weights[4];
		uint32_t bones[4];
	};

	void cuCreateStream(void** ptr);

	void cuDestroyStream(void* ptr);

	template<typename T>
	void cuGetBuffer(T** buf, int size);
	
	template<typename T>
	void cuFree(T* buf);

	bool cuRunBodyUpdate(void* stream, int n, cuVertex* input, cuVector3* output, cuBone* boneData);

	bool cuRunPerVertexUpdate(void* stream, int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData);

	bool cuRunPerTriangleUpdate(void* stream, int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData);

	bool cuSynchronize();
}
