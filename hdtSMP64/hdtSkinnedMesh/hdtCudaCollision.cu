#include "hdtCudaCollision.cuh"

#include "math.h"

// Check collider bounding boxes on the GPU. This reduces the total amount of work, but is bad for
// divergence and increases register usage. Probably no longer very useful with vertex lists working.
//#define GPU_BOUNDING_BOX_CHECK

namespace hdt
{
    // Block size for map type operations (vertex and bounding box calculations). Since there's no reduction
    // in these, we just set this for maximum occupancy (currently 100% for bounding boxes, 75% for vertex
    // calculation).
    constexpr int cuMapBlockSize() { return 128; }

    // Block size for bounding box reduction. Each warp here is independent - larger blocks just do multiple
    // chunks at once. Should be at least 64 for maximum occupancy.
    constexpr int cuReduceBlockSize() { return 64; }

    template<typename T>
    constexpr int collisionBlockSize();

    // Block size for collision checking. Must be a power of 2 for the simple inter-warp reductions to work.
    template<>
    constexpr int collisionBlockSize<cuPerVertexInput>() { return 256; }
    template<>
    constexpr int collisionBlockSize<cuPerTriangleInput>() { return 256; }

    // Maximum number of vertices per patch
    __host__ __device__
    constexpr int vertexListSize() { return 256; }

    // Maximum number of iterations of collision checking with a single vertex list. If there are too many
    // potential collisions to finish in this number of passes, we compute the second vertex list as well.
    __device__
    constexpr int vertexListThresholdFactor() { return 4; }

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

    __device__ cuAabb::cuAabb()
        : aabbMin({ FLT_MAX, FLT_MAX, FLT_MAX, 0 }), aabbMax({ -FLT_MAX, -FLT_MAX, -FLT_MAX, 0 })
    {}

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

    template <unsigned int BlockSize = cuMapBlockSize()>
    __global__ void kernelPerVertexUpdate(int n, const cuPerVertexInput* __restrict__ in, cuAabb* __restrict__ out, const cuVector3* __restrict__ vertexData)
    {
        int index = blockIdx.x * BlockSize + threadIdx.x;
        int stride = BlockSize * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            const cuVector3 v = vertexData[in[i].vertexIndex];
            cuAabb aabb(v);
            aabb.addMargin(v.w * in[i].margin);
            out[i] = aabb;
        }
    }

    template <unsigned int BlockSize = cuMapBlockSize()>
    __global__ void kernelPerTriangleUpdate(int n, const cuPerTriangleInput* __restrict__ in, cuAabb* __restrict__ out, const cuVector3* __restrict__ vertexData)
    {
        int index = blockIdx.x * BlockSize + threadIdx.x;
        int stride = BlockSize * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            const cuVector3 v0 = vertexData[in[i].vertexIndices[0]];
            const cuVector3 v1 = vertexData[in[i].vertexIndices[1]];
            const cuVector3 v2 = vertexData[in[i].vertexIndices[2]];

            float penetration = abs(in[i].penetration);
            float margin = max((v0.w + v1.w + v2.w) * in[i].margin / 3, penetration);

            cuAabb aabb(v0, v1, v2);
            aabb.addMargin(margin);
            out[i] = aabb;
        }
    }

    template< unsigned int BlockSize = cuReduceBlockSize() >
    __global__ void kernelBoundingBoxReduce(int n, const std::pair<int, int>* __restrict__ nodeData, const cuAabb* __restrict__ boundingBoxes, cuAabb* output)
    {
        int tid = threadIdx.x;
        int threadInWarp = tid & 0x1f;
        int warpid = tid >> 5;
        constexpr int nwarps = BlockSize >> 5;
        int stride = gridDim.x * nwarps;

        for (int block = blockIdx.x * nwarps + warpid; block < n; block += stride)
        {
            const cuAabb* aabbStart = boundingBoxes + nodeData[block].first;
            int aabbCount = nodeData[block].second;

            // Load the first block of bounding boxes
            cuAabb temp = (threadInWarp < aabbCount) ? aabbStart[threadInWarp] : cuAabb();

            // Take union with each successive block
            for (int i = threadInWarp + 32; i < aabbCount; i += 32)
            {
                temp.aabbMin = perElementMin(temp.aabbMin, aabbStart[i].aabbMin);
                temp.aabbMax = perElementMax(temp.aabbMax, aabbStart[i].aabbMax);
            }

            // Intra-warp reduce
            for (int j = 16; j > 0; j >>= 1)
            {
                temp.aabbMin.x = min(temp.aabbMin.x, __shfl_down_sync(0xffffffff, temp.aabbMin.x, j));
                temp.aabbMin.y = min(temp.aabbMin.y, __shfl_down_sync(0xffffffff, temp.aabbMin.y, j));
                temp.aabbMin.z = min(temp.aabbMin.z, __shfl_down_sync(0xffffffff, temp.aabbMin.z, j));
                temp.aabbMax.x = max(temp.aabbMax.x, __shfl_down_sync(0xffffffff, temp.aabbMax.x, j));
                temp.aabbMax.y = max(temp.aabbMax.y, __shfl_down_sync(0xffffffff, temp.aabbMax.y, j));
                temp.aabbMax.z = max(temp.aabbMax.z, __shfl_down_sync(0xffffffff, temp.aabbMax.z, j));
            }

            // Store result
            if (threadInWarp == 0)
            {
                output[block].aabbMin = temp.aabbMin;
                output[block].aabbMax = temp.aabbMax;
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

    template <unsigned int BlockSize = cuMapBlockSize()>
    __global__ void kernelBodyUpdate(int n, const cuVertex* __restrict__ in, cuVector3* __restrict__ out, const cuBone* __restrict__ boneData)
    {
        int index = blockIdx.x * BlockSize + threadIdx.x;
        int stride = BlockSize * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            cuVector3 pos = in[i].position;
            cuVector3 v = calcVertexState(pos, boneData[in[i].bones[0]], in[i].weights[0]);
            for (int j = 1; j < 4; ++j)
            {
                v += calcVertexState(pos, boneData[in[i].bones[j]], in[i].weights[j]);
            }
            out[i] = v;
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
        const cuVector3 vA = vertexDataA[inputA.vertexIndex];
        const cuVector3 vB = vertexDataB[inputB.vertexIndex];

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
        float area2 = raw_normal.magnitude2();
        float area = sqrt(area2);

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
        ac = crossProduct(ap, bp);
        ab = crossProduct(cp, ap);
        float areaC = dotProduct(ac, raw_normal);
        float areaB = dotProduct(ab, raw_normal);
        float areaA = area2 - areaB - areaC;
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

    template<int BlockSize>
    __device__ int kernelComputeVertexList(
        int start,
        int n,
        int tid,
        const cuAabb* boundingBoxes,
        const cuAabb& boundingBox,
        int* intShared,
        int* vertexList
    )
    {
        int* partialSums = intShared + 32;
        int threadInWarp = tid & 0x1f;
        int warpid = tid >> 5;
        constexpr int nwarps = BlockSize >> 5;

        // Set up vertex list for shape
        int nCeil = (((n - 1) / BlockSize) + 1) * BlockSize;
        int blockStart = 0;
        for (int i = tid; blockStart < vertexListSize() && i < nCeil; i += BlockSize)
        {
            int vertex = i + start;
            bool collision = i < n && boundingBoxCollision(boundingBoxes[vertex], boundingBox);

            // Count the number of collisions in this warp and store in shared memory
            auto mask = __ballot_sync(0xffffffff, collision);
            if (threadInWarp == 0)
            {
                intShared[warpid] = __popc(mask);
            }
            __syncthreads();

            // Compute partial sum counts for warps
            if (warpid == 0)
            {
                int a = intShared[threadInWarp];
                for (int j = 1; j < nwarps; j <<= 1)
                {
                    int b = __shfl_up_sync(0xffffffff, a, j);
                    if (threadInWarp >= j)
                    {
                        a += b;
                    }
                }
                partialSums[threadInWarp] = blockStart + a;
            }

            __syncthreads();

            // Now we can calculate where to put the index, if it's a potential collision
            if (collision)
            {
                int warpStart = (warpid > 0) ? partialSums[warpid - 1] : blockStart;
                unsigned int lanemask = (1UL << threadInWarp) - 1;
                int index = warpStart + __popc(mask & lanemask);
                if (index < vertexListSize())
                {
                    vertexList[index] = vertex;
                }
            }

            blockStart = partialSums[nwarps - 1];
        }

        // Update number of colliders in A and the total number of pairs
        return min(blockStart, vertexListSize());
    }

    // kernelCollision does the supporting work for threading the collision checks and making sure that only
    // the deepest result is kept.
    template <cuPenetrationType penType = eNone, typename T, int BlockSize = collisionBlockSize<T>()>
    __global__ void __launch_bounds__(collisionBlockSize<T>(), 1024 / collisionBlockSize<T>()) kernelCollision(
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
        __shared__ float floatShared[64 + 2 * vertexListSize()];
        int* intShared = reinterpret_cast<int*>(floatShared);

        int tid = threadIdx.x;
        int threadInWarp = tid & 0x1f;
        int warpid = tid >> 5;
        constexpr int nwarps = BlockSize >> 5;
        
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

            if (nPairs <= BlockSize * vertexListThresholdFactor())
            {
                // If we have fewer possible collider pairs than threads, we can just check every pair in one
                // step, and skip the bounding box check altogether.
                for (int i = tid; i < nPairs; i += BlockSize)
                {
                    int iA = offsetA + i % nA;
                    int iB = offsetB + i / nA;
                    if (collidePair<penType>(inA[iA], inB[iB], vertexDataA, vertexDataB, temp))
                    {
                        temp.colliderA = static_cast<cuCollider*>(0) + iA;
                        temp.colliderB = static_cast<cuCollider*>(0) + iB;
                    }
                }
            }
            else
            {
                int* vertexListA = intShared + 64;
                int* vertexListB = vertexListA + vertexListSize();

                bool haveListA = false;
                bool haveListB = false;

                // Try to get the number of pairs down to a reasonable size, by building a collider list in
                // shared memory from the larger patch. If the result is still large, do the smaller one as
                // well. The threshold for doing this may need some tweaking for optimal performance, but
                // the full collision check is pretty expensive so it won't be more than a couple of
                // blocks.
                if (nA > nB)
                {
                    nA = kernelComputeVertexList<BlockSize>(
                        offsetA,
                        nA,
                        tid,
                        boundingBoxesA,
                        setup[block].boundingBoxB,
                        intShared,
                        vertexListA
                    );
                    haveListA = true;
                    if (nA * nB > BlockSize * vertexListThresholdFactor())
                    {
                        nB = kernelComputeVertexList<BlockSize>(
                            offsetB,
                            nB,
                            tid,
                            boundingBoxesB,
                            setup[block].boundingBoxA,
                            intShared,
                            vertexListB
                        );
                        haveListB = true;
                    }
                }
                else
                {
                    nB = kernelComputeVertexList<BlockSize>(
                        offsetB,
                        nB,
                        tid,
                        boundingBoxesB,
                        setup[block].boundingBoxA,
                        intShared,
                        vertexListB
                    );
                    haveListB = true;
                    if (nA * nB > BlockSize * vertexListThresholdFactor())
                    {
                        nA = kernelComputeVertexList<BlockSize>(
                            offsetA,
                            nA,
                            tid,
                            boundingBoxesA,
                            setup[block].boundingBoxB,
                            intShared,
                            vertexListA
                        );
                        haveListA = true;
                    }
                }

                // kernelComputeVertexList doesn't do a final synchronize, because it's OK to run it
                // sequentially for both lists without synchronizing between them. So we need to synchronize
                // now to make sure the vertex lists are fully visible.
                __syncthreads();

                nPairs = nA * nB;

                for (int i = tid; i < nPairs; i += BlockSize)
                {
                    int iA = haveListA ? vertexListA[i % nA] : offsetA + i % nA;
                    int iB = haveListB ? vertexListB[i / nA] : offsetB + i / nA;

                    // Skip pairs until we find one with a bounding box collision. This should increase the
                    // number of full checks done in parallel, and reduce divergence overall. Note we only do
                    // this at all if there are more pairs than threads - if there's only enough work for a
                    // single iteration (very common), there's no benefit to trying to reduce it.
#ifdef GPU_BOUNDING_BOX_CHECK
                    if (nPairs > BlockSize)
                    {
                        while (i < nPairs && !boundingBoxCollision(boundingBoxesA[iA], boundingBoxesB[iB]))
                        {
                            i += BlockSize;
                            if (i < nPairs)
                            {
                                iA = haveListA ? vertexListA[i % nA] : offsetA + i % nA;
                                iB = haveListB ? vertexListB[i / nA] : offsetB + i / nA;
                            }
                        }
                    }
#endif

                    if (i < nPairs && collidePair<penType>(inA[iA], inB[iB], vertexDataA, vertexDataB, temp))
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
                floatShared[warpid] = d;
            }
            __syncthreads();

            // Find minimum across warps
            if (warpid == 0)
            {
                d = floatShared[threadInWarp];
                for (int j = nwarps >> 1; j > 0; j >>= 1)
                {
                    d = min(d, __shfl_down_sync(0xffffffff, d, j));
                }
                if (threadInWarp == 0)
                {
                    floatShared[0] = d;
                }
            }
            __syncthreads();

            // If the depth of this thread is equal to the minimum, try to set the result. Do an atomic
            // exchange with the first value to ensure that only one thread gets to do this in case of ties.
            if (floatShared[0] == temp.depth && atomicExch(floatShared, 2) == temp.depth)
            {
                output[block] = temp;
            }
        }
    }

    cuResult cuCreateStream(void** ptr)
    {
        *ptr = new cudaStream_t;
        return cudaStreamCreate(reinterpret_cast<cudaStream_t*>(*ptr));
    }

    void cuDestroyStream(void* ptr)
    {
        cudaStreamDestroy(*reinterpret_cast<cudaStream_t*>(ptr));
        delete reinterpret_cast<cudaStream_t*>(ptr);
    }

    cuResult cuGetDeviceBuffer(void** buf, int size)
    {
        return cudaMalloc(buf, size);
    }

    cuResult cuGetHostBuffer(void** buf, int size)
    {
        return cudaMallocHost(buf, size);
    }

    void cuFreeDevice(void* buf)
    {
        cudaFree(buf);
    }

    void cuFreeHost(void* buf)
    {
        cudaFreeHost(buf);
    }

    cuResult cuCopyToDevice(void* dst, void* src, size_t n, void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, *s);
    }

    cuResult cuCopyToHost(void* dst, void* src, size_t n, void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, *s);
    }

    cuResult cuRunBodyUpdate(void* stream, int n, cuVertex* input, cuVector3* output, cuBone* boneData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuMapBlockSize() + 1;

        kernelBodyUpdate <<<numBlocks, cuMapBlockSize(), 0, *s >>> (n, input, output, boneData);
        return cuResult();
    }

    cuResult cuRunPerVertexUpdate(void* stream, int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuMapBlockSize() + 1;

        kernelPerVertexUpdate <<<numBlocks, cuMapBlockSize(), 0, *s >>> (n, input, output, vertexData);
        return cuResult();
    }

    cuResult cuRunPerTriangleUpdate(void* stream, int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        int numBlocks = (n - 1) / cuMapBlockSize() + 1;

        kernelPerTriangleUpdate <<<numBlocks, cuMapBlockSize(), 0, *s >>> (n, input, output, vertexData);
        return cuResult();
    }

    template<cuPenetrationType penType, typename T>
    cuResult cuRunCollision(
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

        kernelCollision<penType> <<<n, collisionBlockSize<T>(), 0, *s >>> (
            n, setup, inA, inB, boundingBoxesA, boundingBoxesB, vertexDataA, vertexDataB, output);
        return cuResult();
    }

    cuResult cuRunBoundingBoxReduce(void* stream, int n, int largestNode, std::pair<int, int>* setup, cuAabb* boundingBoxes, cuAabb* output)
    {
        // Reduction kernel only uses a single warp per tree node, becoming linear performance if there are
        // more than 64 boxes. The reduction itself is entirely intra-warp, without any shared memory use.
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);
        constexpr int warpsPerBlock = cuReduceBlockSize() >> 5;
        int nBlocks = ((n - 1) / warpsPerBlock) + 1;
        kernelBoundingBoxReduce <<<nBlocks, cuReduceBlockSize(), 0, *s >>> (n, setup, boundingBoxes, output);
        return cuResult();
    }

    cuResult cuSynchronize(void* stream)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);

        if (s)
        {
            return cudaStreamSynchronize(*s);
        }
        else
        {
            return cudaDeviceSynchronize();
        }
    }

    cuResult cuCreateEvent(void** ptr)
    {
        *ptr = new cudaEvent_t;
        return cudaEventCreate(reinterpret_cast<cudaEvent_t*>(*ptr));
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

    template cuResult cuRunCollision<eNone, cuPerVertexInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerVertexInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template cuResult cuRunCollision<eNone, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template cuResult cuRunCollision<eExternal, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
    template cuResult cuRunCollision<eInternal, cuPerTriangleInput>(void*, int, cuCollisionSetup*, cuPerVertexInput*, cuPerTriangleInput*, cuAabb*, cuAabb*, cuVector3*, cuVector3*, cuCollisionResult*);
}
