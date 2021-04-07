#include "hdtCudaCollision.cuh"

#include "math.h"

namespace hdt
{
    __device__
        void subtract(const cuVector3& v1, const cuVector3& v2, cuVector3& result)
    {
        result.x = v1.x - v2.x;
        result.y = v1.y - v2.y;
        result.z = v1.z - v2.z;
        result.w = v1.w - v2.w;
    }

    __device__
        void add(const cuVector3& v1, const cuVector3& v2, cuVector3& result)
    {
        result.x = v1.x + v2.x;
        result.y = v1.y + v2.y;
        result.z = v1.z + v2.z;
        result.w = v1.w + v2.w;
    }

    __device__
        void multiply(const cuVector3& v, float c, cuVector3& result)
    {
        result.x = v.x * c;
        result.y = v.y * c;
        result.z = v.z * c;
        result.w = v.w * c;
    }

    __device__
        void crossProduct(const cuVector3& v1, const cuVector3& v2, cuVector3& result)
    {
        result.x = v1.y * v2.z - v1.z * v2.y;
        result.y = v1.z * v2.x - v1.x * v2.z;
        result.z = v1.x * v2.y - v1.y * v2.x;
    }

    __device__
        void normalize(cuVector3& v)
    {
        float mag = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        v.x /= mag;
        v.y /= mag;
        v.z /= mag;
    }

    __global__
        void kernelPerVertexUpdate(int n, const cuPerVertexInput* __restrict__ in, cuAabb* __restrict__ out, const cuVector3* __restrict__ vertexData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            const cuVector3& v = vertexData[in[i].vertexIndex];
            float margin = v.w * in[i].margin;

            out[i].aabbMin.x = v.x - margin;
            out[i].aabbMin.y = v.y - margin;
            out[i].aabbMin.z = v.z - margin;
            out[i].aabbMax.x = v.x + margin;
            out[i].aabbMax.y = v.y + margin;
            out[i].aabbMax.z = v.z + margin;
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

            out[i].aabbMin.x = min(v0.x, min(v1.x, v2.x)) - margin;
            out[i].aabbMin.y = min(v0.y, min(v1.y, v2.y)) - margin;
            out[i].aabbMin.z = min(v0.z, min(v1.z, v2.z)) - margin;
            out[i].aabbMax.x = max(v0.x, max(v1.x, v2.x)) + margin;
            out[i].aabbMax.y = max(v0.y, max(v1.y, v2.y)) + margin;
            out[i].aabbMax.z = max(v0.z, max(v1.z, v2.z)) + margin;
        }
    }

    __device__ cuVector3& operator+=(cuVector3& v1, cuVector3& v2)
    {
        v1.x += v2.x;
        v1.y += v2.y;
        v1.z += v2.z;
        v1.w += v2.w;
        return v1;
    }

    __device__ cuVector3 calcVertexState(const cuVector3& skinPos, const cuBone& bone, float w)
    {
        cuVector3 result;
        result.x = bone.transform[0].x * skinPos.x + bone.transform[1].x * skinPos.y + bone.transform[2].x * skinPos.z + bone.transform[3].x;
        result.y = bone.transform[0].y * skinPos.x + bone.transform[1].y * skinPos.y + bone.transform[2].y * skinPos.z + bone.transform[3].y;
        result.z = bone.transform[0].z * skinPos.x + bone.transform[1].z * skinPos.y + bone.transform[2].z * skinPos.z + bone.transform[3].z;
        result.w = bone.marginMultiplier.w;
        result.x *= w;
        result.y *= w;
        result.z *= w;
        result.w *= w;
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

    __global__
        void kernelVertexVertexCollision(
            int n,
            const cuCollisionSetup* __restrict__ setup,
            cuCollisionResult* output)
    {
        extern __shared__ float sdata[];

        for (int block = blockIdx.x; block < n; block += gridDim.x)
        {
            int nA = setup[block].sizeA;
            int nB = setup[block].sizeB;
            const cuPerVertexInput* __restrict__ inA = setup[block].colliderBufA;
            const cuPerVertexInput* __restrict__ inB = setup[block].colliderBufB;
            const cuVector3* __restrict__ vertexDataA = setup[block].vertexDataA;
            const cuVector3* __restrict__ vertexDataB = setup[block].vertexDataB;

            // Depth should always be negative for collisions. Initialize it to 1 to signify no collision
            // detected.
            int tid = threadIdx.x;
            cuCollisionResult temp;
            temp.depth = 1;

            // First process each block, and keep the deepest one in temp
            for (int i = tid; i < nA * nB; i += blockDim.x)
            {
                const cuPerVertexInput& inputA = inA[i % nA];
                const cuPerVertexInput& inputB = inB[i / nA];
                const cuVector3& vA = vertexDataA[inputA.vertexIndex];
                const cuVector3& vB = vertexDataB[inputB.vertexIndex];

                float rA = vA.w * inputA.margin;
                float rB = vB.w * inputA.margin; // FIXME: Suspect this ought to be inputB, but the original code was this way
                float bound2 = (rA + rB) * (rA + rB);
                cuVector3 diff;
                subtract(vA, vB, diff);
                float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                float len = sqrt(dist2);
                float dist = len - (rA + rB);
                if (dist2 <= bound2 && (dist < temp.depth))
                {
                    if (len <= FLT_EPSILON)
                    {
                        diff = { 1, 0, 0, 0 };
                    }
                    else
                    {
                        normalize(diff);
                    }
                    temp.depth = dist;
                    temp.normOnB = diff;
                    multiply(diff, rA, temp.posA);
                    multiply(diff, rB, temp.posB);
                    subtract(vA, temp.posA, temp.posA);
                    add(vB, temp.posB, temp.posB);
                    temp.colliderA = static_cast<cuCollider*>(0) + i % nA;
                    temp.colliderB = static_cast<cuCollider*>(0) + i / nA;
                }
            }

            // Set the best depth in shared data
            sdata[tid] = temp.depth;

            // Now do a reduce operation so we end up with the minimum depth in the first element
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s && sdata[tid] > sdata[tid + s])
                {
                    sdata[tid] = sdata[tid + s];
                }
                __syncthreads();
            }

            // If our depth is equal to the minimum, set the result. Atomic exchange ensures that only one
            // thread can do this even if there are several with the same depth.
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

    bool cuRunCollision(void* stream, int n, cuCollisionSetup* setup, cuCollisionResult* output)
    {
        cudaStream_t* s = reinterpret_cast<cudaStream_t*>(stream);

        kernelVertexVertexCollision <<<n, cuBlockSize(), cuBlockSize() * sizeof(float), *s >>> (n, setup, output);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool cuRunCollision(void* stream, int nA, int nB, cuPerVertexInput* inA, cuPerTriangleInput* inB, cuCollisionResult* output, cuVector3* vertexDataA, cuVector3* vertexDataB)
    {
        return true;
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
}
