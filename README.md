# hdtSMP for Skyrim VR (not working yet)

Fork of [https://github.com/Karonar1/hdtSMP64] by Karonar1, of fork of [version](https://github.com/aers/hdtSMP64) by aers, from
[original code](https://github.com/HydrogensaysHDT/hdt-skyrimse-mods) by hydrogensaysHDT

## Changes 

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
  physics should now work (with limitations) on NPCs without having to manually edit the facegen data.
+ New bones from facegen files should now be added to the head instead of the NPC root, so they should now be
  positioned correctly if there is no physics for them or after a reset.

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

These should be complete instructions to set up and build the plugin, without assuming prior coding
experience. I'm checking everything out into D:\Dev-noAVX and building without AVX support, but of course you
can use any directory.

You will need:
+ Visual Studio 2019 (any edition)
+ Git
+ CMake

Open a VS2019 command prompt ("**x64 Native Tools Command Prompt for VS2019**"). Download and build Detours and
Bullet:

```
d:
mkdir Dev-noAVX
cd Dev-noAVX
git clone https://github.com/microsoft/Detours.git
git clone https://github.com/bulletphysics/bullet3.git
cd Detours
nmake
cd ..\bullet3
cmake .
```

If you want AVX support in Bullet, use `cmake-gui` instead of `cmake`, and check the `USE_MSVC_AVX` box
before clicking Configure then Generate. This should give a fairly significant performance boost if your CPU
supports it.
+ Note that AVX support in Bullet and the HDT-SMP plugin itself are independent configuration options. Enable
  it in both for maximum performance; disable it in both for maximum compatibility.

Open D:\Dev-noAVX\bullet3\BULLET_PHYSICS.sln in Visual Studio, select the Release configuration, then 
Build -> Build solution.

Download https://skse.silverlock.org/beta/sksevr_2_00_12.7z and unpack into Dev-noAVX (source code is included in the official distribution),
then get the HDT-SMP source:

```
cd D:\Dev-noAVX\sksevr_2_00_12\src\sksevr
git init
git remote add origin https://github.com/alandtse/hdtSMP64.git
git fetch
git checkout master
```

Open D:\Dev-noAVX\sksevr_2_00_12\src\sksevr\hdtSMP64.sln in Visual Studio. If you are asked to retarget
projects, just accept the defaults and click OK.

Open properties for the hdtSMP64 project. Select "All Configurations" at the top, and the C/C++ page. Add the
following to Additional Include Directories (just delete anything that was there before):
+ D:\Dev-noAVX\Detours\include
+ D:\Dev-noAVX\bullet3\src
+ D:\Dev-noAVX\sksevr_2_00_12\src

On the Linker -> General page, add the following to Additional Library Directories:
+ D:\Dev-noAVX\bullet3\lib\Release
+ D:\Dev-noAVX\Detours\lib.X64

Open properties for the skse64 project. Select General, and change Configuration Type from "Dynamic Library
(.dll)" to "Static Library (.lib)".

Make the following changes to the SKSE code:

In NiObjects.h, at line 81, delete:
```cpp
virtual void			* Unk_05(void);
```

and replace it with:
```cpp
virtual BSFadeNode		* GetAsBSFadeNode(void);
```

At line 88, delete:
```cpp
virtual void			* Unk_0C(void);
```

and replace it with:
```cpp
virtual BSDynamicTriShape	* GetAsBSDynamicTriShape(void);
```

In the same file, at line 36 (the end of the list of class declarations), add:
```cpp
class BSDynamicTriShape;
class BSFadeNode;
```

And at line 174 (inside the `enum` definition), add:
```cpp
kNone =		0,
```

In GameEvents.h, at line 667, delete:
```cpp
EventDispatcher<void>								unk840;					//  840 - sink offset 0C8
```

and replace it with:
```cpp
EventDispatcher<TESMoveAttachDetachEvent>			unk840;					//  840 - sink offset 0C8
```

In GameMenus.h, befor line 1105 (just before the GetSingleton declaration), add:
```cpp
bool IsGamePaused() { return numPauseGame > 0; }
```

Now you should be able to select the Release or Release_noAVX configuration and build the plugin.

## Credits

hydrogensaysHDT - Creating this plugin

aers - fixes and improvements

ousnius - some fixes, and "consulting"


