#pragma once

#include <cstdint>

namespace hdt
{
	// signatures generated with IDA SigMaker plugin for SSE
	// comment is SSE, uncommented is VR. VR address found by Ghidra compare using known SSE offset to find same function 
	// (based on function signature and logic)
	namespace offset
	{

#ifndef SKYRIMVR
		// hdtSkyrimPhysicsWorld.cpp
		// 74 35 45 33 C0 33 D2
		constexpr std::uintptr_t GameStepTimer_SlowTime = 0x02F6B948;

		// Hooks.cpp
		// E8 ? ? ? ? 48 8B E8 FF C7
		constexpr std::uintptr_t ArmorAttachFunction = 0x001CAFB0;


		// BSFaceGenNiNode last vfunc
		constexpr std::uintptr_t BSFaceGenNiNode_SkinAllGeometry = 0x003D87B0;
		constexpr std::uintptr_t BSFaceGenNiNode_SkinSingleGeometry = 0x003D8840;


		// Hooks.cpp
		// function responsible for majority of main game thread loop
		// E8 ? ? ? ? 84 DB 74 24 
		constexpr std::uintptr_t GameLoopFunction = 0x005B2FF0;

		// E8 ? ? ? ? E8 ? ? ? ? E8 ? ? ? ? 48 8B 0D ? ? ? ? 48 85 C9 74 0C E8 ? ? ? ? 
		constexpr std::uintptr_t GameShutdownFunction = 0x01293D20;


		// FaceGeom string
		constexpr std::uintptr_t TESNPC_GetFaceGeomPath = 0x00363210;


		// BSFaceGenModelExtraData Bone Limit
		// 8B 70 58 EB 02 
		constexpr std::uintptr_t BSFaceGenModelExtraData_BoneLimit = 0x0036B4C8;

#else
		constexpr std::uintptr_t GameStepTimer_SlowTime = 0x030C3A08;
		constexpr std::uintptr_t ArmorAttachFunction = 0x001DB9E0;
		constexpr std::uintptr_t BSFaceGenNiNode_SkinAllGeometry = 0x003e8120;
		constexpr std::uintptr_t BSFaceGenNiNode_SkinSingleGeometry = 0x003e81b0;
		constexpr std::uintptr_t GameLoopFunction = 0x005BAB10;
		constexpr std::uintptr_t GameShutdownFunction = 0x012CC630;
		constexpr std::uintptr_t TESNPC_GetFaceGeomPath = 0x000372b30;
		constexpr std::uintptr_t BSFaceGenModelExtraData_BoneLimit = 0x00037ae28;
#endif
		// .text:00000001403D88D4                 cmp     ebx, 8
		// patch 8 -> 7
		//VR is same as SSE
		constexpr std::uintptr_t BSFaceGenNiNode_SkinSingleGeometry_bug = BSFaceGenNiNode_SkinSingleGeometry + 0x96;
	}
}
