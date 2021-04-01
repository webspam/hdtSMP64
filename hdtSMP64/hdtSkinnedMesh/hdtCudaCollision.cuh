#pragma once

namespace hdt
{
	struct cuVector3
	{
		float x;
		float y;
		float z;
		float w;
	};

	struct cuTriangle
	{
		cuVector3 pA;
		cuVector3 pB;
		cuVector3 pC;
	};

	struct cuPerVertexInput
	{
		cuVector3 point;
		float margin;
	};

	struct cuPerVertexOutput
	{
		cuVector3 aabbMin;
		cuVector3 aabbMax;
	};

	template<typename T>
	void cuGetBuffer(T** buf, int size);
	
	template<typename T>
	void cuFree(T* buf);

	bool cuRunPerVertexUpdate(int n, cuPerVertexInput* input, cuPerVertexOutput* output);
}
