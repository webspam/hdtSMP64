#include "hdtFrameTimer.h"

#include "skse64/GameAPI.h"

namespace hdt
{
	FrameTimer* FrameTimer::instance()
	{
		static FrameTimer s_instance;
		return &s_instance;
	}

	void FrameTimer::reset(int nFrames)
	{
		m_nFrames = nFrames;
		m_count = nFrames / 2;
		m_sumsCPU.clear();
		m_sumsSquaredCPU.clear();
		m_sumsGPU.clear();
		m_sumsSquaredGPU.clear();
		m_nManifoldsCPU = 0;
		m_nManifolds2CPU = 0;
		m_nManifoldsGPU = 0;
		m_nManifolds2GPU = 0;
	}

	void FrameTimer::logEvent(FrameTimer::Events e)
	{
		if (!running())
		{
			return;
		}

		LARGE_INTEGER ticks;
		QueryPerformanceCounter(&ticks);
		m_timings[e] = ticks.QuadPart;

		if (e == e_End)
		{
			QueryPerformanceFrequency(&ticks);
			float ticks_per_us = static_cast<float>(ticks.QuadPart) / 1e6;
			float internalTime = (m_timings[e_Internal] - m_timings[e_Start]) / ticks_per_us;
			float collisionTime = (m_timings[e_End] - m_timings[e_Internal]) / ticks_per_us;
			float totalTime = (m_timings[e_End] - m_timings[e_Start]) / ticks_per_us;

			if (cudaFrame())
			{
				m_sumsGPU[e_InternalUpdate] += internalTime;
				m_sumsSquaredGPU[e_InternalUpdate] += internalTime * internalTime;
				m_sumsGPU[e_Collision] += collisionTime;
				m_sumsSquaredGPU[e_Collision] += collisionTime * collisionTime;
				m_sumsGPU[e_Total] += totalTime;
				m_sumsSquaredGPU[e_Total] += totalTime * totalTime;
			}
			else
			{
				m_sumsCPU[e_InternalUpdate] += internalTime;
				m_sumsSquaredCPU[e_InternalUpdate] += internalTime * internalTime;
				m_sumsCPU[e_Collision] += collisionTime;
				m_sumsSquaredCPU[e_Collision] += collisionTime * collisionTime;
				m_sumsCPU[e_Total] += totalTime;
				m_sumsSquaredCPU[e_Total] += totalTime * totalTime;
			}

			if (--m_nFrames == 0)
			{
				Console_Print("Timings over %d frames:", m_count);
				Console_Print("  CPU:");
				float mean = m_sumsCPU[e_InternalUpdate] / m_count;
				Console_Print("    Internal update mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredCPU[e_InternalUpdate] / m_count - mean * mean));
				mean = m_sumsCPU[e_Collision] / m_count;
				Console_Print("    Collision check mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredCPU[e_Collision] / m_count - mean * mean));
				mean = m_sumsCPU[e_Total] / m_count;
				Console_Print("    Total mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredCPU[e_Total] / m_count - mean * mean));
				mean = m_nManifoldsCPU / m_count;
				Console_Print("    Collision manifolds %f, std %f",
					mean,
					sqrt(m_nManifolds2CPU / m_count - mean * mean));

				Console_Print("  GPU:");
				mean = m_sumsGPU[e_InternalUpdate] / m_count;
				Console_Print("    Internal update mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredGPU[e_InternalUpdate] / m_count - mean * mean));
				mean = m_sumsGPU[e_Collision] / m_count;
				Console_Print("    Collision check mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredGPU[e_Collision] / m_count - mean * mean));
				mean = m_sumsGPU[e_Total] / m_count;
				Console_Print("    Total mean %f us, std %f us",
					mean,
					sqrt(m_sumsSquaredGPU[e_Total] / m_count - mean * mean));
				mean = m_nManifoldsGPU / m_count;
				Console_Print("    Collision manifolds %f, std %f",
					mean,
					sqrt(m_nManifolds2GPU / m_count - mean * mean));
			}
		}
	}

	void FrameTimer::addManifoldCount(int nManifolds)
	{
		if (cudaFrame())
		{
			m_nManifoldsGPU += nManifolds;
			m_nManifolds2GPU += nManifolds * nManifolds;
		}
		else
		{
			m_nManifoldsCPU += nManifolds;
			m_nManifolds2CPU += nManifolds * nManifolds;
		}
	}

	bool FrameTimer::running()
	{
		return m_nFrames > 0;
	}

	bool FrameTimer::cudaFrame()
	{
		return m_nFrames > m_count;
	}
}
