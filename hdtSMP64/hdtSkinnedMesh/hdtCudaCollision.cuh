#pragma once

#include <memory>
#include <immintrin.h>

#include <tuple>
#include <string>

#ifndef __NVCC__
#define __host__
#define __device__
#define __forceinline__ __forceinline
#endif

#include "hdtCudaPlanarStruct.cuh"

namespace hdt
{
	class CudaPerVertexShape;
	class CudaPerTriangleShape;

	class cuResult
	{
	public:

#ifdef __NVCC__
		cuResult(cudaError_t error = cudaGetLastError())
			: m_ok(error == cudaSuccess)
		{
			if (!m_ok)
			{
				m_message = cudaGetErrorString(error);
			}
		}
#else
		bool check(std::string context)
		{
			if (!m_ok)
			{
				_MESSAGE("%s: %s", context.c_str(), m_message.c_str());
			}
			return m_ok;
		}
#endif

		operator bool()
		{
			return m_ok;
		}

		std::string message()
		{
			return m_message;
		}

	private:

		bool m_ok;
		std::string m_message;
	};

	// Internal penetration is the most common case where penetration is positive
	// No penetration is where it is zero, and the triangle is treated as unoriented
	// If penetration is negative, we handle it at the interface level by swapping two vertices and negating
	// penetration, making it equivalent to the internal case.
	enum cuPenetrationType
	{
		eInternal,
		eNone
	};

	struct cuVector3;

	union cuVector4
	{
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		__m128 val;

#ifdef __NVCC__
		__device__ cuVector4();
		__device__ cuVector4(float ix, float iy, float iz, float iw);
		__device__ cuVector4(const cuVector3& v);
		__device__ explicit operator cuVector3() const;
		__device__ float* __restrict__ vals() { return &x; }
		__device__ const float* __restrict__ vals() const { return &x; }

		__device__ cuVector4 operator+(const cuVector4& o) const;
		__device__ cuVector4 operator-(const cuVector4& o) const;
		__device__ cuVector4 operator*(const float c) const;

		__device__ cuVector4& operator+=(const cuVector4& o);
		__device__ cuVector4& operator-=(const cuVector4& o);
		__device__ cuVector4& operator*=(const float c);

		__device__ float magnitude2() const;
		__device__ float magnitude() const;
		__device__ cuVector4 normalize() const;
#endif
	};

	struct cuVector3
	{
#ifdef __NVCC__
		__device__ cuVector3(float ix, float iy, float iz);
#endif
		float x;
		float y;
		float z;
	};

	struct cuTriangleIndices
	{
		int a;
		int b;
		int c;
	};

	struct cuPerVertexInput
	{
		int vertexIndex;
		float margin;
	};

	struct cuPerTriangleInput
	{
		cuTriangleIndices vertexIndices;
		float margin;
		float penetration;
	};

	struct cuAabb3
	{
#ifdef __NVCC__
		__device__ cuAabb3();
		__device__ cuAabb3(const cuVector3& mins, const cuVector3& maxs);
		__device__ explicit cuAabb3(const cuVector4& v);

		template<typename... Args>
		__device__ explicit cuAabb3(const cuVector4& v, const Args&... args);

		__device__ void addMargin(const float margin);
#endif

		cuVector3 aabbMin;
		cuVector3 aabbMax;
	};

	struct cuAabb
	{
		cuVector4 aabbMin;
		cuVector4 aabbMax;
	};

	struct cuBone
	{
		cuVector4 transform[4];
		cuVector4 marginMultiplier; // Dummy, no longer used
#ifdef __NVCC__
		__device__ const float* __restrict__ vals() const { return &transform[0].x; }
#endif
	};

	struct cuVertex
	{
		cuVector4 position;
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
		cuVector4 posA;
		cuVector4 posB;
		cuVector4 normOnB;
		int colliderA;
		int colliderB;
		float depth;
	};

	struct cuCollisionSetup
	{
		int sizeA;
		int sizeB;
		int offsetA;
		int offsetB;
		cuAabb boundingBoxA;
		cuAabb boundingBoxB;
	};

	struct cuCollisionMerge
	{
		cuVector4 normal;
		cuVector4 posA;
		cuVector4 posB;
		float weight;
	};

	using VectorArray = ArrayType<cuVector3, float, float, float>;
	using BoundingBoxArray = ArrayType<cuAabb3, VectorArray, VectorArray>;
	using VertexInputArray = ArrayType<cuPerVertexInput, int, float>;
	using TriangleInputArray = ArrayType<cuPerTriangleInput, ArrayType<cuTriangleIndices, int, int, int>, float, float>;

	cuResult cuCreateStream(void** ptr);

	void cuDestroyStream(void* ptr);

	cuResult cuGetDeviceBuffer(void** buf, int size);

	cuResult cuGetHostBuffer(void** buf, int size);

	void cuFreeDevice(void* buf);

	void cuFreeHost(void* buf);

	cuResult cuCopyToDevice(void* dst, void* src, size_t n, void* stream);

	cuResult cuCopyToHost(void* dst, void* src, size_t n, void* stream);

	cuResult cuMemset(void* buf, int value, size_t n, void* stream);

	template <cuPenetrationType penType = eNone, typename T>
	cuResult cuRunCollision(
		void* stream,
		int n,
		bool swap,
		cuCollisionSetup* setup,
		VertexInputArray inA,
		T inB,
		BoundingBoxArray boundingBoxesA,
		BoundingBoxArray boundingBoxesB,
		cuVertex* vertexSetupA,
		cuVertex* vertexSetupB,
		cuVector4* vertexDataA,
		cuVector4* vertexDataB,
		float* boneWeightsA,
		float* boneWeightsB,
		int* boneMapA,
		int* boneMapB,
		cuCollisionMerge* mergeBuffer,
		int mergeX,
		int mergeDynX,
		int mergeY);

	cuResult cuInternalUpdate(
		void* stream,
		int nVertices,
		const cuVertex* verticesIn,
		cuVector4* vertexData,
		const cuBone* boneData,
		int nVertexColliders,
		VertexInputArray perVertexIn,
		BoundingBoxArray perVertexOut,
		int nVertexNodes,
		const std::pair<int, int>* vertexNodeData,
		cuAabb* vertexNodeOutput,
		int nTriangleColliders,
		TriangleInputArray perTriangleIn,
		BoundingBoxArray perTriangleOut,
		int nTriangleNodes,
		const std::pair<int, int>* triangleNodeData,
		cuAabb* triangleNodeOutput);

	cuResult cuSynchronize(void* stream = nullptr);

	cuResult cuCreateEvent(void** ptr);

	void cuDestroyEvent(void* ptr);

	void cuRecordEvent(void* ptr, void* stream);

	void cuWaitEvent(void* ptr);

	void cuInitialize();

	int cuDeviceCount();

	void cuSetDevice(int id);

	int cuGetDevice();
}
