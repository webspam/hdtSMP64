#include "hdtCudaCollision.cuh"

#include "math.h"

// Check collider bounding boxes on the GPU. This reduces the total amount of work, but is bad for
// divergence and increases register usage.
#define GPU_BOUNDING_BOX_CHECK

namespace hdt
{
    constexpr int cuBlockSize() { return 1024; }

    // Reduction makes heavy use of shared memory (16 bytes per thread), so a smaller block size may be desirable
    constexpr int cuReduceBlockSize() { return 1024; }

    template<typename T>
    constexpr int collisionBlockSize();

    // Collision checking is quite register-hungry, so we may need to reduce the block size for it here
    template<>
    constexpr int collisionBlockSize<cuPerVertexInput>() { return 1024; }
    template<>
    constexpr int collisionBlockSize<cuPerTriangleInput>() { return 1024; }

    __device__ cuVector3::cuVector3()
    {}

    __device__ __forceinline__ cuVector3::cuVector3(float ix, float iy, float iz, float iw)
        : x(ix), y(iy), z(iz), w(iw)
    {}

    __device__ __forceinline__ cuVector3 cuVector3::operator+(const cuVector3& o) const
    {
        return { x + o.x, y + o.y, z + o.z, w + o.w };
    }

    __device__ __forceinline__ cuVector3 cuVector3::operator-(const cuVector3& o) const
    {
        return { x - o.x, y - o.y, z - o.z, w - o.w };
    }

    __device__ __forceinline__ cuVector3 cuVector3::operator*(const float c) const
    {
        return { x * c, y * c, z * c, w * c };
    }

    __device__ __forceinline__ cuVector3& cuVector3::operator+=(const cuVector3& o)
    {
        *this = *this + o;
        return *this;
    }

    __device__ __forceinline__ cuVector3& cuVector3::operator-=(const cuVector3& o)
    {
        *this = *this - o;
        return *this;
    }

    __device__ __forceinline__ cuVector3& cuVector3::operator *= (const float c)
    {
        *this = *this * c;
        return *this;
    }

    __device__
        cuVector3 crossProduct(const cuVector3& v1, const cuVector3& v2)
    {
        return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x, 0 };
    }

    __device__
        float dotProduct(const cuVector3& v1, const cuVector3& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    __device__ float cuVector3::magnitude2() const
    {
        return dotProduct(*this, *this);
    }

    __device__ float cuVector3::magnitude() const
    {
        return sqrt(magnitude2());
    }

    __device__ cuVector3 cuVector3::normalize() const
    {
        return *this * rsqrt(magnitude2());
    }

    __device__ __forceinline__ cuVector3 perElementMin(const cuVector3& v1, const cuVector3& v2)
    {
        return { min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z), min(v1.w, v2.w) };
    }

    __device__ __forceinline__ cuVector3 perElementMax(const cuVector3& v1, const cuVector3& v2)
    {
        return { max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z), max(v1.w, v2.w) };
    }

    __device__ cuAabb::cuAabb(const cuVector3& v)
        : aabbMin(v), aabbMax(v)
    {}

    template<typename... Args>
    __device__ __forceinline__ cuAabb::cuAabb(const cuVector3& v, const Args&... args)
        : cuAabb(args...)
    {
        aabbMin = perElementMin(aabbMin, v);
        aabbMax = perElementMax(aabbMax, v);
    }

    __device__ void cuAabb::addMargin(const float margin)
    {
        aabbMin.x -= margin;
        aabbMin.y -= margin;
        aabbMin.z -= margin;
        aabbMax.x += margin;
        aabbMax.y += margin;
        aabbMax.z += margin;
    }

    __device__
        bool boundingBoxCollision(const cuAabb& b1, const cuAabb& b2)
    {
        return !(b1.aabbMin.x > b2.aabbMax.x ||
            b1.aabbMin.y > b2.aabbMax.y ||
            b1.aabbMin.z > b2.aabbMax.z ||
            b1.aabbMax.x < b2.aabbMin.x ||
            b1.aabbMax.y < b2.aabbMin.y ||
            b1.aabbMax.z < b2.aabbMin.z);
    }

    __global__
        void kernelPerVertexUpdate(int n, const cuPerVertexInput* __restrict__ in, cuAabb* __restrict__ out, const cuVector3* __restrict__ vertexData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            const cuVector3& v = vertexData[in[i].vertexIndex];
            out[i] = cuAabb(v);
            out[i].addMargin(v.w * in[i].margin);
        }
    }

    __global__
        void kernelPerTriangleUpdate(int n, const cuPerTriangleInput* __restrict__ in, cuAabb* __restrict__ out, const cuVector3* __restrict__ vertexData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            const cuVector3& v0 = vertexData[in[i].vertexIndices[0]];
            const cuVector3& v1 = vertexData[in[i].vertexIndices[1]];
            const cuVector3& v2 = vertexData[in[i].vertexIndices[2]];

            float penetration = abs(in[i].penetration);
            float margin = max((v0.w + v1.w + v2.w) * in[i].margin / 3, penetration);

            out[i] = cuAabb(v0, v1, v2);
            out[i].addMargin(margin);
        }
    }

    __global__
        void kernelBoundingBoxReduce(int n, const std::pair<int, int>* __restrict__ nodeData, const cuAabb* __restrict__ boundingBoxes, cuAabb* output)
    {
        extern __shared__ cuVector3 shared[];

        for (int block = blockIdx.x; block < n; block += gridDim.x)
        {
            const cuAabb* aabbStart = boundingBoxes + nodeData[block].first;
            int aabbCount = nodeData[block].second;
            int tid = threadIdx.x;

            // To reduce demand on shared memory, process min and max separately. We may also need to process
            // the data in multiple blocks - we use the fast divide-and-conquer approach within a block, but
            // combine them linearly.
            cuVector3 temp = { FLT_MAX, FLT_MAX, FLT_MAX, 0 };
            for (int i = tid; i < aabbCount; i += 2 * blockDim.x)
            {
                // First step takes data from the individual bounding boxes and populates shared memory
                int s = blockDim.x;
                if (i + s < aabbCount)
                {
                    shared[tid] = perElementMin(aabbStart[i].aabbMin, aabbStart[i + s].aabbMin);
                }
                else
                {
                    shared[tid] = aabbStart[i].aabbMin;
                }

                // Now we can do a conventional reduction
                s >>= 1;
                __syncthreads();
                for (; s > 0; s >>= 1)
                {
                    if (tid < s && i + s < aabbCount)
                    {
                        shared[tid] = perElementMin(shared[tid], shared[tid + s]);
                    }
                    __syncthreads();
                }

                // Finally, thread 0 combines with the result from previous blocks
                if (tid == 0)
                {
                    temp = perElementMin(temp, shared[tid]);
                }
            }
            if (tid == 0)
            {
                output[block].aabbMin = temp;
            }

            // Now do the same again for the maximums
            temp = { -FLT_MAX, -FLT_MAX, -FLT_MAX, 0 };
            for (int i = tid; i < aabbCount; i += 2 * blockDim.x)
            {
                int s = blockDim.x;
                if (i + s < aabbCount)
                {
                    shared[tid] = perElementMax(aabbStart[i].aabbMax, aabbStart[i + s].aabbMax);
                }
                else
                {
                    shared[tid] = aabbStart[i].aabbMin;
                }
                s >>= 1;
                __syncthreads();
                for (; s > 0; s >>= 1)
                {
                    if (tid < s && i + s < aabbCount)
                    {
                        shared[tid] = perElementMax(shared[tid], shared[tid + s]);
                    }
                    __syncthreads();
                }
                if (tid == 0)
                {
                    temp = perElementMax(temp, shared[tid]);
                }
            }
            if (tid == 0)
            {
                output[block].aabbMax = temp;
            }
        }
    }

    __device__ cuVector3 calcVertexState(const cuVector3& skinPos, const cuBone& bone, float w)
    {
        cuVector3 result;
        result.x = bone.transform[0].x * skinPos.x + bone.transform[1].x * skinPos.y + bone.transform[2].x * skinPos.z + bone.transform[3].x;
        result.y = bone.transform[0].y * skinPos.x + bone.transform[1].y * skinPos.y + bone.transform[2].y * skinPos.z + bone.transform[3].y;
        result.z = bone.transform[0].z * skinPos.x + bone.transform[1].z * skinPos.y + bone.transform[2].z * skinPos.z + bone.transform[3].z;
        result.w = bone.marginMultiplier.w;
        result *= w;
        return result;
    }

    __global__
        void kernelBodyUpdate(int n, const cuVertex* __restrict__ in, cuVector3* __restrict__ out, const cuBone* __restrict__ boneData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            out[i] = calcVertexState(in[i].position, boneData[in[i].bones[0]], in[i].weights[0]);
            for (int j = 1; j < 4; ++j)
            {
                out[i] += calcVertexState(in[i].position, boneData[in[i].bones[j]], in[i].weights[j]);
            }
        }
    }

    // collidePair does the actual collision between two colliders, always a vertex and some other type. It
    // should modify output if and only if there is a collision.
    template <cuPenetrationType>
    __device__ bool collidePair(
        const cuPerVertexInput& __restrict__ inputA,
        const cuPerVertexInput& __restrict__ inputB,
        const cuVector3* __restrict__ vertexDataA,
        const cuVector3* __restrict__ vertexDataB,
        cuCollisionResult& output)
    {
        const cuVector3& vA = vertexDataA[inputA.vertexIndex];
        const cuVector3& vB = vertexDataB[inputB.vertexIndex];

        float rA = vA.w * inputA.margin;
        float rB = vB.w * inputB.margin;
        float bound2 = (rA + rB) * (rA + rB);
        cuVector3 diff = vA - vB;
        float dist2 = diff.magnitude2();
        float len = sqrt(dist2);
        float dist = len - (rA + rB);
        if (dist2 <= bound2 && (dist < output.depth))
        {
            if (len <= FLT_EPSILON)
            {
                diff = { 1, 0, 0, 0 };
            }
            else
            {
                diff = diff.normalize();
            }
            output.depth = dist;
            output.normOnB = diff;
            output.posA = vA - diff * rA;
            output.posB = vB + diff * rB;
            return true;
        }
        return false;
    }

    template <cuPenetrationType penType>
    __device__ bool collidePair(
        const cuPerVertexInput& __restrict__ inputA,
        const cuPerTriangleInput& __restrict__ inputB,
        const cuVector3* __restrict__ vertexDataA,
        const cuVector3* __restrict__ vertexDataB,
        cuCollisionResult& output)
    {
        cuVector3 s = vertexDataA[inputA.vertexIndex];
        float r = s.w * inputA.margin;
        cuVector3 p0 = vertexDataB[inputB.vertexIndices[0]];
        cuVector3 p1 = vertexDataB[inputB.vertexIndices[1]];
        cuVector3 p2 = vertexDataB[inputB.vertexIndices[2]];
        float margin = (p0.w + p1.w + p2.w) / 3.0;
        float penetration = inputB.penetration * margin;
        margin *= inputB.margin;

        // Compute unit normal and twice area of triangle
        cuVector3 ab = p1 - p0;
        cuVector3 ac = p2 - p0;
        cuVector3 raw_normal = penType == eExternal ? crossProduct(ac, ab) : crossProduct(ab, ac);
        float area = raw_normal.magnitude();

        // Check for degenerate triangles
        if (area < FLT_EPSILON)
        {
            return false;
        }
        cuVector3 normal = raw_normal * (1.0 / area);

        // Compute distance from point to plane and its projection onto the plane
        cuVector3 ap = s - p0;
        float distance = dotProduct(ap, normal);

        float radiusWithMargin = r + margin;
        if (penType == eNone)
        {
            // Two-sided check: make sure distance is positive and normal is in the correct direction
            if (distance < 0)
            {
                distance = -distance;
                normal *= -1.0;
            }
        }
        else if (distance < penetration)
        {
            // One-sided check: make sure sphere center isn't too far on the wrong side of the triangle
            return false;
        }

        // Don't bother to do any more if there's no collision or we already have a deeper one
        float depth = distance - radiusWithMargin;
        if (depth >= -FLT_EPSILON || depth >= output.depth)
        {
            return false;
        }

        // Compute triple products and check the projection lies in the triangle
        cuVector3 bp = s - p1;
        cuVector3 cp = s - p2;
        cuVector3 aa = crossProduct(bp, cp);
        ab = crossProduct(cp, ap);
        ac = crossProduct(ap, bp);
        float areaA = dotProduct(aa, raw_normal);
        float areaB = dotProduct(ab, raw_normal);
        float areaC = dotProduct(ac, raw_normal);
        if (areaA < 0 || areaB < 0 || areaC < 0)
        {
            return false;
        }

        output.normOnB = normal;
        output.posA = s - normal * r;
        output.posB = s - normal * (distance - margin);
        output.depth = depth;
        return true;
    }

    // kernelCollision does the supporting work for threading the collision checks and making sure that only
    // the deepest result is kept.
    template <cuPenetrationType penType = eNone, typename T>
    __global__ void kernelCollision(
        int n,
        const cuCollisionSetup* __restrict__ setup,
        const cuPerVertexInput* __restrict__ inA,
        const T* __restrict__ inB,
        const cuAabb* __restrict__ boundingBoxesA,
        const cuAabb* __restrict__ boundingBoxesB,
        const cuVector3* __restrict__ vertexDataA,
        const cuVector3* __restrict__ vertexDataB,
        cuCollisionResult* output)
    {
        extern __shared__ float sdata[];
        int* intShared = reinterpret_cast<int*>(sdata);

        int tid = threadIdx.x;
        int threadInWarp = tid & 0x1f;
        int warpid = threadIdx.x >> 5;
        
        for (int block = blockIdx.x; block < n; block += gridDim.x)
        {
            int nA = setup[block].sizeA;
            int nB = setup[block].sizeB;
            int offsetA = setup[block].offsetA;
            int offsetB = setup[block].offsetB;

            // Depth should always be negative for collisions. We'll use positive values to signify no
            // collision, and later for mutual exclusion.
            cuCollisionResult temp;
            temp.depth = 1;
            int nPairs = nA * nB;

            if (nPairs < blockDim.x)
            {
                // If we have fewer possible collider pairs than threads, we can just check every pair in one
                // step, and skip the bounding box check altogether.
                if (tid < nPairs)
                {
                    int iA = offsetA + tid % nA;
                    int iB = offsetB +  tid / nA;
                    if (collidePair<penType>(inA[iA], inB[iB], vertexDataA, vertexDataB, temp))
                    {
                        temp.colliderA = static_cast<cuCollider*>(0) + iA;
                        temp.colliderB = static_cast<cuCollider*>(0) + iB;
                    }
                }
            }
            else
            {
                int* indicesA = setup[block].scratch;
                const cuAabb& boundingBoxB = setup[block].boundingBoxB;

                // Set up vertex list for shape A
                int nACeil = (((nA - 1) / blockDim.x) + 1) * blockDim.x;
                int blockStart = 0;
                for (int i = tid; i < nACeil; i += blockDim.x)
                {
                    int vertex = i + offsetA;
                    bool collision = i < nA&& boundingBoxCollision(boundingBoxesA[vertex], boundingBoxB);

                    // Count the number of collisions in this warp and store in shared memory
                    auto mask = __ballot_sync(0xffffffff, collision);
                    if (threadInWarp == 0)
                    {
                        intShared[warpid] = __popc(mask);
                    }
                    __syncthreads();

                    // Compute partial sum counts in warps before this one and broadcast across the warp
                    int a = (threadInWarp < warpid) ? intShared[threadInWarp] : 0;
                    for (int j = 16; j > 0; j >>= 1)
                    {
                        a += __shfl_down_sync(0xffffffff, a, j);
                    }
                    int warpStart = blockStart + __shfl_sync(0xffffffff, a, 0);

                    // Now we can calculate where to put the index, if it's a potential collision
                    if (collision)
                    {
                        int lanemask = (1L << threadInWarp) - 1;
                        indicesA[warpStart + __popc(mask & lanemask)] = vertex;
                    }

                    // Extend the partial sum from the last warp to a total sum, and update the block start
                    if (warpid == 31 && threadInWarp == 0)
                    {
                        intShared[0] = a + intShared[31];
                    }
                    __syncthreads();
                    blockStart += intShared[0];
                }

                // Update number of colliders in A and the total number of pairs
                nA = blockStart;
                nPairs = nA * nB;

                for (int i = tid; i < nPairs; i += blockDim.x)
                {
                    int iA = indicesA[i % nA];
                    int iB = offsetB + i / nA;

                    // Skip pairs until we find one with a bounding box collision. This should increase the
                    // number of full checks done in parallel, and reduce divergence overall. Note we only do
                    // this at all if there are more pairs than threads - if there's only enough work for a
                    // single iteration (very common), there's no benefit to trying to reduce it.
#ifdef GPU_BOUNDING_BOX_CHECK
                    if (nPairs > blockDim.x)
                    {
                        while (i < nPairs && !boundingBoxCollision(boundingBoxesA[iA], boundingBoxesB[iB]))
                        {
                            i += blockDim.x;
                            iA = indicesA[i % nA];
                            iB = offsetB + i / nA;
                        }
                    }
#endif

                    if (i < nPairs&& collidePair<penType>(inA[iA], inB[iB], vertexDataA, vertexDataB, temp))
                    {
                        temp.colliderA = static_cast<cuCollider*>(0) + iA;
                        temp.colliderB = static_cast<cuCollider*>(0) + iB;
                    }
                }
            }

            // Find minimum depth in this warp and store in shared memory
            float d = temp.depth;
            for (int j = 16; j > 0; j >>= 1)
            {
                d = min(d, __shfl_down_sync(0xffffffff, d, j));
            }
            if (threadInWarp == 0)
            {
                sdata[warpid] = d;
            }
            __syncthreads();

            // Find minimum across warps
            if (warpid == 0)
            {
                d = sdata[threadInWarp];
                for (int j = 16; j > 0; j >>= 1)
                {
                    d = min(d, __shfl_down_sync(0xffffffff, d, j));
                }
                if (threadInWarp == 0)
                {
                    sdata[0] = d;
                }
            }
            __syncthreads();

            // If the depth of this thread is equal to the minimum, try to set the result. Do an atomic
            // exchange with the first value to ensure that only one thread gets to do this in case of ties.
            if (sdata[0] == temp.depth && atomicExch(sdata, 2) == temp.depth)
            {
                output[block] = temp;
            }
        }
    }

    void cuCreateStream(void** ptr)
    {
        *ptr = new cudaStream_t;
        cudaStreamCreate(reinterpret_cast<cudaStream_t*>(*ptr));
    }

    void cuDestroyStream(void* ptr)
    {
        cudaStreamDestroy(*reinterpret_cast<cudaStream_t*>(ptr));
        delete reinterpret_cast<cudaStream_t*>(ptr);
    }

    void cuGetDeviceBuffer(void** buf, int size)
    {
        cudaMalloc(buf, size);
    }

    void cuGetHostBuffer(void** buf, int size)
    {
        cudaMallocHost(buf, size);
    }

    void cuFreeDevice(void* buf)
    {
        cudaFree(buf);
    }

    void cuFreeHost(void* buf)
    {
        cudaFreeHost(buf);
    }

    void cuCopyToDevice(void* dst, void* src, size_t n, void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, *s);
    }

    void cuCopyToHost(void* dst, void* src, size_t n, void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, *s);
    }

    bool cuRunBodyUpdate(void* stream, int n, cuVertex* input, cuVector3* output, cuBone* boneData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuBlockSize() + 1;

        kernelBodyUpdate <<<numBlocks, cuBlockSize(), 0, *s >>> (n, input, output, boneData);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool cuRunPerVertexUpdate(void* stream, int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuBlockSize() + 1;

        kernelPerVertexUpdate <<<numBlocks, cuBlockSize(), 0, *s >>> (n, input, output, vertexData);
        return cudaPeekAtLastError() == cudaSuccess;
    }


    bool cuRunPerTriangleUpdate(void* stream, int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuBlockSize() + 1;

        kernelPerTriangleUpdate <<<numBlocks, cuBlockSize(), 0, *s >>> (n, input, output, vertexData);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    template<cuPenetrationType penType, typename T>
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
        cuCollisionResult* output)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);

        kernelCollision<penType> <<<n, collisionBlockSize<T>(), 32 * sizeof(float), *s >>> (
            n, setup, inA, inB, boundingBoxesA, boundingBoxesB, vertexDataA, vertexDataB, output);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool cuRunBoundingBoxReduce(void* stream, int n, int largestNode, std::pair<int, int>* setup, cuAabb* boundingBoxes, cuAabb* output)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);

        // Block size for bounding box reduction should be half the size of the largest block, rounded up to
        // a multiple of 32, not exceeding the maximum block size.
        int blockSize = min(32 * ((largestNode - 1) / 64 + 1), cuReduceBlockSize());
        
        kernelBoundingBoxReduce <<<n, blockSize, blockSize * sizeof(cuVector3), *s >>> (n, setup, boundingBoxes, output);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool cuSynchronize(void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);

        if (s)
        {
            return cudaStreamSynchronize(*s);
        }
        else
        {
            return cudaDeviceSynchronize() == cudaSuccess;
        }
    }

    void cuCreateEvent(void** ptr)
    {
        *ptr = new cudaEvent_t;
        cudaEventCreate(reinterpret_cast<cudaEvent_t*>(*ptr));
    }

    void cuDestroyEvent(void* ptr)
    {
        cudaEventDestroy(*reinterpret_cast<cudaEvent_t*>(ptr));
        delete reinterpret_cast<cudaEvent_t*>(ptr);
    }

    void cuRecordEvent(void* ptr, void* stream)
    {
        cudaEvent_t* e = reinterpret_cast<cudaEvent_t*>(ptr);
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        cudaEventRecord(*e, *s);
    }

    void cuWaitEvent(void* ptr)
    {
        cudaEvent_t* e = reinterpret_cast<cudaEvent_t*>(ptr);
        cudaEventSynchronize(*e);
    }

    void cuInitialize()
    {
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
    }

    int cuDeviceCount()
    {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }

    template bool cuRunCollision<eNone, cuPerVertexInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerVertexInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template bool cuRunCollision<eNone, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template bool cuRunCollision<eExternal, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template bool cuRunCollision<eInternal, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
}
