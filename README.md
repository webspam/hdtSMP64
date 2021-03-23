# hdtSMP for Skyrim Special Edition

Fork of [version](https://github.com/aers/hdtSMP64) by aers, from
[original code](https://github.com/HydrogensaysHDT/hdt-skyrimse-mods) by hydrogensaysHDT

## Changes 

+ Added distance check in ActorManager to disable NPCs more than a certain distance from the player. This
  resolves the massive FPS drop in certain cell transitions (such as Blue Palace -> Solitude).
+ Added can-collide-with-bone to mesh definitions, as a natural counterpart to no-collide-with-bone.
+ Added new "external" sharing option, for objects that should collide with other NPCs, but not this one.
+ Significant refactoring of armor handling in ActorManager, to be much stricter about disabling systems and
  reducing gradual FPS loss. Added workaround for armors with the same prefix, which previously remained
  permanently attached and could cause slowdowns or crashes.
+ SkyrimBone now keeps a proper reference to its underlying node, which prevents a bug when an NPC with
  active physics had their equipment changed. This was most noticeable when changing costumes on characters
  with short SMP hairs.
+ Skeletons should remain active as long as they are in the same cell as (and close enough to) the player
  character. Resolves an issue where entering the Ancestor Glade often incorrectly marked skeletons as
  inactive and disabled physics.

## Coming soon (maybe)

+ Reworked tag system for better compartmentalized .xml files.
+ More parallelism on collision checking.

## Known issues

+ Several options, including shape and collision definitions on bones, exist but don't seem to do anything.
+ Sphere-triangle collision check without penetration defined is obviously wrong, but fixing the test
  doesn't improve things. Needs further investigation.

## Build Instructions

Requires Visual Studio 2019 (Community Edition is fine)

1) Download the clean skse64 source from [skse's website](http://skse.silverlock.org/)
2) Extract or clone this repository into the skse64 folder of the source. You can safely delete skse64_loader, skse64_steam_loader, skse64_loader_common, and the skse64 solution.
3) Download and build [bullet physics](https://github.com/bulletphysics/bullet3) using cmake and Visual Studio. Update the hdtSMP64 project to point to your bullet source and built libraries. The Release configuration is intended to be built with Bullet's "Use MSVC AVX" cmake option enabled, and Release_noavx without.
4) Download and build [Microsoft's detours library](https://github.com/microsoft/Detours) using nmake. Update the hdtSMP64 project to point to your built library.
5) Make the edits to SKSE source described below.
6) Build the solution :)

## SKSE Edits

GameMenus.h line 1090: 

add function 
```cpp
	bool                IsGamePaused() { return numPauseGame > 0; }
```
to MenuManager class

GameEvents.h line 662:

replace unk840 with
```cpp
EventDispatcher<TESMoveAttachDetachEvent>    		unk840;					//  840 - sink offset 0C8
```

NiObjects.h around line 207:

replace 

```cpp
	float		unkF8;				// 0F8
	UInt32		unkFC;				// 0FC
```

with

```cpp
	TESObjectREFR* m_owner; // 0F8
```

You will need to add a forward declaration or include for TESObjectREFR in NiObjects.h as well.

## Credits

hydrogensaysHDT - Creating this plugin

aers - fixes and improvements

ousnius - some fixes, and "consulting"


