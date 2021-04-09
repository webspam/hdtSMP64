#pragma once

#include <memory>
#include <immintrin.h>

namespace hdt
{
	class CudaPerVertexShape;
	class CudaPerTriangleShape;

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

	// Never actually used, just need its size for setting results
	struct alignas(16) cuCollider
	{
		int vertexIndices[3];
		float flexible;
	};

	struct cuCollisionResult
	{
		cuVector3 posA;
		cuVector3 posB;
		cuVector3 normOnB;
		cuCollider* colliderA;
		cuCollider* colliderB;
		float depth;
	};

	template<typename T>
	struct cuCollisionSetup;

	template<>
	struct cuCollisionSetup<CudaPerVertexShape>
	{
		int sizeA;
		int sizeB;
		cuPerVertexInput* colliderBufA;
		cuPerVertexInput* colliderBufB;
		cuAabb* boundingBoxesA;
		cuAabb* boundingBoxesB;
		cuVector3* vertexDataA;
		cuVector3* vertexDataB;
	};

	template<>
	struct cuCollisionSetup<CudaPerTriangleShape>
	{
		int sizeA;
		int sizeB;
		cuPerVertexInput* colliderBufA;
		cuPerTriangleInput* colliderBufB;
		cuAabb* boundingBoxesA;
		cuAabb* boundingBoxesB;
		cuVector3* vertexDataA;
		cuVector3* vertexDataB;
	};

	void cuCreateStream(void** ptr);

	void cuDestroyStream(void* ptr);

	void cuGetDeviceBuffer(void** buf, int size);

	void cuGetHostBuffer(void** buf, int size);

	void cuFreeDevice(void* buf);

	void cuFreeHost(void* buf);

	void cuCopyToDevice(void* dst, void* src, size_t n, void* stream);

	void cuCopyToHost(void* dst, void* src, size_t n, void* stream);

	bool cuRunBodyUpdate(void* stream, int n, cuVertex* input, cuVector3* output, cuBone* boneData);

	bool cuRunPerVertexUpdate(void* stream, int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData);

	bool cuRunPerTriangleUpdate(void* stream, int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData);

	template <typename T>
	bool cuRunCollision(void* stream, int n, cuCollisionSetup<T>* setup, cuCollisionResult* output);

	bool cuSynchronize(void* stream = nullptr);

	void cuCreateEvent(void** ptr);

	void cuDestroyEvent(void* ptr);

	void cuRecordEvent(void* ptr, void* stream);

	void cuWaitEvent(void* ptr);

	void cuInitialize();
}
