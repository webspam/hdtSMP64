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

	template <typename StructT, typename... Args>
	struct PlanarStruct
	{
	private:

		template<typename T, typename... Ts>
		friend struct PlanarStruct;

		using TupleT = std::tuple<Args...>;

		template<typename T>
		struct ElementSize
		{
			static const size_t size = sizeof(T);
		};
		template<typename... Ts>
		struct ElementSize<PlanarStruct<Ts...>>
		{
			static const size_t size = PlanarStruct<Ts...>::size;
		};
		template<int N, typename... Args>
		struct OffsetHelper;
		template<typename... Args>
		struct OffsetHelper<0, Args...>
		{
			static const size_t offset = 0;
		};
		template<int N, typename First, typename... Rest>
		struct OffsetHelper<N, First, Rest...>
		{
			static const size_t offset = ElementSize<First>::size + OffsetHelper<N - 1, Rest...>::offset;
		};
		template<int N>
		static const size_t offset = OffsetHelper<N, Args...>::offset;
		static const size_t size = OffsetHelper<sizeof...(Args), Args...>::offset;

	public:

		__host__ __device__ PlanarStruct(StructT* buffer, size_t n)
			: m_buffer(reinterpret_cast<uint8_t*>(buffer)), m_size(n)
		{
			static_assert(size == sizeof(StructT), "Element sizes do not match");
		}

		template <int N>
		__device__ __forceinline__ auto getPlane()
		{
			return reinterpret_cast<typename std::tuple_element<N, TupleT>::type*>(
				m_buffer + m_size * offset<N>);
		}

		struct GetHelper
		{
		private:
			template <typename T, typename... Ts>
			friend struct PlanarStruct;

			template <int N, typename T>
			struct Getter
			{
				using type = T&;
				using value_type = T;

				__device__ __forceinline__ static type get(PlanarStruct* s, size_t n)
				{
					return s->getPlane<N>()[n];
				}
			};
			template <int N, typename T, typename... Ts>
			struct Getter<N, PlanarStruct<T, Ts...>>
			{
				using type = typename PlanarStruct<T, Ts...>::GetHelper;
				using value_type = T;

				__device__ __forceinline__ static type get(PlanarStruct* s, size_t n)
				{
					return PlanarStruct<T, Ts...>(reinterpret_cast<T*>(s->getPlane<N>()), s->m_size)[n];
				}
			};

		public:

			template <int N>
			__device__ __forceinline__ auto get()
				-> typename Getter<N, typename std::tuple_element<N, TupleT>::type>::type
			{
				return Getter<N, typename std::tuple_element<N, TupleT>::type>::get(m_s, m_n);
			}

			template <int N>
			__device__ __forceinline__ const auto get() const
			{
				return Getter<N, typename std::tuple_element<N, TupleT>::type>::get(m_s, m_n);
			}

			__device__ __forceinline__ operator StructT()
			{
				return getStruct(std::make_index_sequence<sizeof...(Args)>());
			}

			__device__ __forceinline__ operator const StructT() const
			{
				return getStruct(std::make_index_sequence<sizeof...(Args)>());
			}

			__device__ __forceinline__ auto operator=(const StructT& s)
			{
				set(s, std::make_index_sequence<sizeof...(Args)>());
				return *this;
			}

		private:
			__device__ __forceinline__ GetHelper(PlanarStruct* s, size_t n)
				: m_s(s), m_n(n) {}

			template <std::size_t... I>
			__device__ __forceinline__ StructT getStruct(std::index_sequence<I...>)
			{
				return StructT(get<I>()...);
			}

			template <std::size_t... I>
			__device__ __forceinline__ const StructT getStruct(std::index_sequence<I...>) const
			{
				return StructT(get<I>()...);
			}

			template<typename... Args>
			__device__ __forceinline__ void dummy(Args...) {}

			template <std::size_t... I>
			__device__ __forceinline__ void set(const StructT& s, std::index_sequence<I...>)
			{
				dummy((get<I>() = *reinterpret_cast<const typename Getter<I, Args>::value_type*>(reinterpret_cast<const uint8_t*>(&s) + offset<I>))...);
			}

			PlanarStruct* m_s;
			size_t m_n;
		};

		__device__ __forceinline__ GetHelper operator[](size_t i)
		{
			return GetHelper(this, i);
		}

		__device__ __forceinline__ const GetHelper operator[](size_t i) const
		{
			return GetHelper(const_cast<PlanarStruct*>(this), i);
		}

	private:

		size_t m_size;
		uint8_t* m_buffer;
	};

	// Internal penetration is the most common case where penetration is positive
	// External penetration is where penetration is negative, and the orientation of the triangle is reversed
	// No penetration is where it is zero, and the triangle is treated as unoriented
	enum cuPenetrationType
	{
		eInternal,
		eExternal,
		eNone
	};

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

	struct cuTriangle
	{
		cuVector4 pA;
		cuVector4 pB;
		cuVector4 pC;
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
		__device__ cuAabb();
		__device__ explicit cuAabb(const cuVector4& v);

		template<typename... Args>
		__device__ explicit cuAabb(const cuVector4& v, const Args&... args);

		__device__ void addMargin(const float margin);
#endif

		cuVector4 aabbMin;
		cuVector4 aabbMax;
	};

	struct cuBone
	{
		cuVector4 transform[4];
		cuVector4 marginMultiplier; // Note only w component actually used
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
		cuAabb boundingBoxA;
		cuAabb boundingBoxB;
	};

	using PlanarVectorArray = PlanarStruct<cuVector4, float, float, float, float>;
	using PlanarBoundingBoxArray = PlanarStruct<cuAabb, PlanarVectorArray, PlanarVectorArray>;

	cuResult cuCreateStream(void** ptr);

	void cuDestroyStream(void* ptr);

	cuResult cuGetDeviceBuffer(void** buf, int size);

	cuResult cuGetHostBuffer(void** buf, int size);

	void cuFreeDevice(void* buf);

	void cuFreeHost(void* buf);

	cuResult cuCopyToDevice(void* dst, void* src, size_t n, void* stream);

	cuResult cuCopyToHost(void* dst, void* src, size_t n, void* stream);

	cuResult cuRunBodyUpdate(void* stream, int n, cuVertex* input, cuVector4* output, cuBone* boneData);

	cuResult cuRunPerVertexUpdate(void* stream, int n, cuPerVertexInput* input, PlanarBoundingBoxArray output, cuVector4* vertexData);

	cuResult cuRunPerTriangleUpdate(void* stream, int n, cuPerTriangleInput* input, PlanarBoundingBoxArray output, cuVector4* vertexData);

	template <cuPenetrationType penType = eNone, typename T>
	cuResult cuRunCollision(
		void* stream,
		int n,
		cuCollisionSetup* setup,
		cuPerVertexInput* inA,
		T* inB,
		PlanarBoundingBoxArray boundingBoxesA,
		PlanarBoundingBoxArray boundingBoxesB,
		cuVector4* vertexDataA,
		cuVector4* vertexDataB,
		cuCollisionResult* output);

	cuResult cuRunBoundingBoxReduce(void* stream, int n, std::pair<int, int>* setup, PlanarBoundingBoxArray boundingBoxes, cuAabb* output);

	cuResult cuSynchronize(void* stream = nullptr);

	cuResult cuCreateEvent(void** ptr);

	void cuDestroyEvent(void* ptr);

	void cuRecordEvent(void* ptr, void* stream);

	void cuWaitEvent(void* ptr);

	void cuInitialize();

	int cuDeviceCount();
}
