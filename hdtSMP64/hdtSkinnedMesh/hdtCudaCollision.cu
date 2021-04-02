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
        void kernelPerVertexUpdate(int n, cuPerVertexInput* in, cuAabb* out, cuVector3* vertexData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            cuVector3& v = vertexData[in[i].vertexIndex];
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
        void kernelPerTriangleUpdate(int n, cuPerTriangleInput* in, cuAabb* out, cuVector3* vertexData)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            cuVector3& v0 = vertexData[in[i].vertexIndices[0]];
            cuVector3& v1 = vertexData[in[i].vertexIndices[1]];
            cuVector3& v2 = vertexData[in[i].vertexIndices[2]];

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

    template<typename T>
    void cuGetBuffer(T** buf, int size)
    {
        cudaMallocManaged(buf, size * sizeof(T));
    }

    template<typename T>
    void cuFree(T* buf)
    {
        cudaFree(buf);
    }

    bool cuRunPerVertexUpdate(int n, cuPerVertexInput* input, cuAabb* output, cuVector3* vertexData)
    {
        int numBlocks = (n - 1) / 512 + 1;

        kernelPerVertexUpdate << <numBlocks, 512 >> > (n, input, output, vertexData);
        return cudaPeekAtLastError() == cudaSuccess;
    }


    bool cuRunPerTriangleUpdate(int n, cuPerTriangleInput* input, cuAabb* output, cuVector3* vertexData)
    {
        int numBlocks = (n - 1) / 512 + 1;

        kernelPerTriangleUpdate << <numBlocks, 512 >> > (n, input, output, vertexData);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool cuSynchronize()
    {
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    template void cuGetBuffer<cuVector3>(cuVector3**, int);
    template void cuGetBuffer<cuPerVertexInput>(cuPerVertexInput**, int);
    template void cuGetBuffer<cuPerTriangleInput>(cuPerTriangleInput**, int);
    template void cuGetBuffer<cuAabb>(cuAabb**, int);

    template void cuFree<cuVector3>(cuVector3*);
    template void cuFree<cuPerVertexInput>(cuPerVertexInput*);
    template void cuFree<cuPerTriangleInput>(cuPerTriangleInput*);
    template void cuFree<cuAabb>(cuAabb*);
}
