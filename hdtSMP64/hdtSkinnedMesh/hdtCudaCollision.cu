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
        void kernelPerVertexUpdate(int n, cuPerVertexInput* in, cuPerVertexOutput* out)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < n; i += stride)
        {
            float margin = in[i].point.w * in[i].margin;
            out[i].aabbMin.x = in[i].point.x - margin;
            out[i].aabbMin.y = in[i].point.y - margin;
            out[i].aabbMin.z = in[i].point.z - margin;
            out[i].aabbMax.x = in[i].point.x + margin;
            out[i].aabbMax.y = in[i].point.y + margin;
            out[i].aabbMax.z = in[i].point.z + margin;
        }
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

    bool cuRunPerVertexUpdate(int n, cuPerVertexInput* input, cuPerVertexOutput* output)
    {
        int numBlocks = (n - 1) / 512 + 1;

        kernelPerVertexUpdate << <numBlocks, 512 >> > (n, input, output);
        return cudaPeekAtLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }

    template void cuGetBuffer<cuPerVertexInput>(cuPerVertexInput**, int);
    template void cuGetBuffer<cuPerVertexOutput>(cuPerVertexOutput**, int);

    template void cuFree<cuPerVertexInput>(cuPerVertexInput*);
    template void cuFree<cuPerVertexOutput>(cuPerVertexOutput*);
}