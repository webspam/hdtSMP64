#pragma once

#include <memory>
#include <immintrin.h>

namespace hdt
{
	class CudaPerVertexShape;
	class CudaPerTriangleShape;

	// Internal penetration is the most common case where penetration is positive
	// External penetration is where penetration is negative, and the orientation of the triangle is reversed
	// No penetration is where it is zero, and the triangle is treated as unoriented
	enum cuPenetrationType
	{
		eInternal,
		eExternal,
		eNone
	};

	union cuVector3
	{
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		__m128 val;

#ifdef __NVCC__
		__device__ cuVector3();
		__device__ cuVector3(float ix, float iy, float iz, float iw);

		__device__ cuVector3 operator+(const cuVector3& o) const;
		__device__ cuVector3 operator-(const cuVector3& o) const;
		__device__ cuVector3 operator*(const float c) const;

		__device__ cuVector3& operator+=(const cuVector3& o);
		__device__ cuVector3& operator-=(const cuVector3& o);
		__device__ cuVector3& operator*=(const float c);

		__device__ float magnitude2() const;
		__device__ float magnitude() const;
		__device__ cuVector3 normalize() const;
#endif
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
#ifdef __NVCC__
		__device__ explicit cuAabb(const cuVector3& v);

		template<typename... Args>
		__device__ explicit cuAabb(const cuVector3& v, const Args&... args);

		__device__ void addMargin(const float margin);
#endif

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

	struct cuCollisionSetup
	{
		int sizeA;
		int sizeB;
		int offsetA;
		int offsetB;
		int* scratch;
		cuAabb boundingBoxB;
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

	template <cuPenetrationType penType = eNone, typename T>
	bool cuRunCollision(
		void* stream,
		int n,
		cuCollisionSetup* setup,
		cuPerVertexInput* inA,
		T* inB,
		cuAabb* boundingBoxesA,
		cuAabb* boundingBoxesB,
		cuVector3* vertexDataA,
		cuVector3* vertexDataB,
		cuCollisionResult* output);

	bool cuRunBoundingBoxReduce(void* stream, int n, int largestNode, std::pair<int, int>* setup, cuAabb* boundingBoxes, cuAabb* output);

	bool cuSynchronize(void* stream = nullptr);

	void cuCreateEvent(void** ptr);

	void cuDestroyEvent(void* ptr);

	void cuRecordEvent(void* ptr, void* stream);

	void cuWaitEvent(void* ptr);

	void cuInitialize();

	int cuDeviceCount();
}
