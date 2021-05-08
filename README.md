# hdtSMP with CUDA for Skyrim Special Edition

Fork of [version](https://github.com/aers/hdtSMP64) by aers, from
[original code](https://github.com/HydrogensaysHDT/hdt-skyrimse-mods) by hydrogensaysHDT

## Changes 

+ Added CUDA support for several parts of collision detection (still a work in progress). This includes
  everything that had OpenCL support in earlier releases, as well as the final collision check.
+ Added distance check in ActorManager to disable NPCs more than a certain distance from the player. This
  resolves the massive FPS drop in certain cell transitions (such as Blue Palace -> Solitude). Default
  maximum distance is 10000, which resolves that issue, but I recommend something around 2000 for better
  general performance in cities.
+ Added can-collide-with-bone to mesh definitions, as a natural counterpart to no-collide-with-bone.
+ Added new "external" sharing option, for objects that should collide with other NPCs, but not this one.
  Good for defining the whole body as a collision mesh for interactions between characters.
+ Significant refactoring of armor handling in ActorManager, to be much stricter about disabling systems and
  reducing gradual FPS loss.
+ Changed prefix mechanism to use a simple incrementing value instead of trying to use pointers to objects.
  Previously this could lead to prefix collisions with a variety of weird effects and occasional crashes.
+ Skeletons should remain active as long as they are in the same cell as (and close enough to) the player
  character. Resolves an issue where entering the Ancestor Glade often incorrectly marked skeletons as
  inactive and disabled physics.
+ Added "smp list" console command to list tracked NPCs without so much detail - useful for checking which
  NPCs are active in crowded areas. NPCs are now sorted according to active status, with active ones last.
+ New mechanism for remapping mesh names in the defaultBBPs.xml file, allowing much more concise ways of
  defining complex collision objects for lots of armors at once.
+ The code to scan defaultBBPs.xml can now handle the structure of facegen files, which means head part
  physics should work (with limitations) on NPCs without having to manually edit the facegen data.
+ New bones from facegen files should now be added to the head instead of the NPC root, so they should be
  positioned correctly if there is no physics for them or after a reset.

## CUDA support

CUDA support is disabled by default, but can be enabled in configs.xml or from the console. It will
automatically fall back to the CPU algorithm if you do not have any CUDA capable cards. However, it does not
check capabilities of any cards it finds, so may crash if your card is too old. It was developed for a
GeForce 10 series card, so should work on those or anything newer.

The following parts of the collision algortihm are currently GPU accelerated:

* Vertex position calculations for skinned mesh bodies
* Collider bounding box calculations for per-vertex and per-triangle shapes
* Aggregate bounding box calculation for internal nodes of collider trees
* Building collider lists for the final collision check (for one body only)
* Sphere-sphere and sphere-triangle collision checks

The following parts are still CPU-based:

* Merging collision results and converting to manifolds for the Bullet library to work with
* And, of course, the solver itself, which is part of the Bullet library, not the HDT-SMP plugin

This is still experimental, and may not give good performance. The old CPU collision algorithm was heavily
optimized, so matching its framerate is not easy.

* On a 6850K processor (6 cores, 3.6GHz) with a 1080Ti GPU, framerate in crowded areas is significantly worse
  than with the CPU-only algorithm. The GPU algorithm typically gives about half the framerate of the CPU
  one when under heavy load. But most of the time, both algorithms easily reach the framerate cap at 60fps.
* On the same hardware, the internal update (vertex position and bounding box calculations) takes about the
  same time on CPU and GPU, typically around 3ms per frame. It's the main collision check algorithm and/or
  its supporting code that kills performance.

If you have an i3 or i5 CPU (or the AMD equivalent) with a fast graphics card, the GPU algorithm may help. If
you have an i7 or i9 CPU, or your graphics card already struggles with the base game, stick with the CPU
version.

## Radeon support?

Nope, sorry. CUDA and nVidia cards are pretty much the industry standard for scientific computing, so that's
what I use. In any case, I can't support GPU architectures that I don't have.

## Note about NPC head parts

Head parts work fine for NPCs without valid facegen data, but this isn't very useful because it triggers the
infamous dark face bug. Special restrictions apply to NPCs that do have facegen data:

+ Only one XML file can be used per face, even if was built from multiple physics-enabled parts.
+ NiStringExtraData nodes aren't automatically copied into the facegen file, so physics won't work
  automatically. Either do it manually in NifSkope or map one of the head parts to a file in defaultBBPs.xml.
+ Bones that aren't explicitly referenced by any mesh are removed when facegen data is generated, and can't
  be used as kinematic objects in constraints. Replace references to these with the NPC head (which should
  always be present). You may also need to set the frame of reference origin to the position of the missing
  bone relative to the head to get correct constraint behavior.

## Console commands

The smp console command will print some basic information about the number of tracked and active objects. The
plugin recognizes the following optional parameters:

* reset attempts to reload all meshes and reset the whole HDT-SMP system. However, it is a little buggy and
  may fail to reload some meshes or constraints properly.
* gpu toggles the CUDA collision algorithm, if there is at least one CUDA device available.
* dumptree dumps the entire node tree of the current targeted NPC to the log file.
* detail shows extended details of all tracked actors, including active and inactive armour and head parts.
* list shows a more concise list of tracked actors.

## Coming soon (maybe)

+ Reworked tag system for better compartmentalized .xml files.
+ More parallelism on collision checking.

## Known issues

+ Several options, including shape and collision definitions on bones, exist but don't seem to do anything.
+ Sphere-triangle collision check without penetration defined is obviously wrong, but fixing the test
  doesn't improve things. Needs further investigation.
+ Smp reset doesn't reload meshes correctly for some items (observed with Artesian Cloaks of Skyrim).
  Suspect references to the original triangle meshes are being dropped when they're no longer needed. We
  could keep ownership of the meshes, but it seems pretty marginal and a waste of memory. It also breaks
  some constraints for NPCs with facegen data.
+ It's not possible to refer to bones in facegen data that aren't used explicitly by at least one mesh. Most
  HDT hairs won't work as non-wig hair on NPCs without altering constraints. Probably possible but annoying
  to fix.
+ Probably any open bug listed on Nexus that isn't resolved in changes, above. This list only contains
  issues I have personally observed.

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


