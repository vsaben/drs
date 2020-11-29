This repository supports the autonomous extraction and dense-annotation of simulated vehicle accident scenarios, as viewed from an aerial perspective, from Grand Theft Auto V (GTA). The data are used to train a modified, YOLO-based 2-stage: 

1. **Object detector**: Far-view vehicle localisation, pose estimation and damage status classification.  
2. **Instance segmentation model**: Close-view identification of specified vehicle damage types.

Aerial vehicle and vehicle damage (VAVD) underly the operation of a novel drone-based infrastructure for responding to and recording urban vehicle accidents (UVAs), hereon referred to as a Drone Response Service (DRS). All coding scripts are provided as an addendum to the thesis titled: *"Vehicle accident feature extraction from a drone-based video feed"*. 

### Drone Response Service (DRS) 

Upon the occurrrence of a vehicle accident, an initiated drone:

1. Autonomously navigates to the accident site.
2. Records visual evidence of the damaged vehicles or property.
3. Relays this information to insurance providers and other interested parties.   

The focus is to develop the computer vision tools to support this operation.

### Machine Learning

YOLO-based networks are used to generate region proposals for a multi-task network head. The head encompasses 5 tasks, computed in parallel: 

### Simulating accidents in GTA

#### Instructions

##### Installation

1. Download 'ScriptHookV.dll' and 'dinput8.dll' ("Download" >> Unzip) from http://www.dev-c.com/gtav/scripthookv/
2. Download 'ScriptHookVDotNet2.dll', 'ScriptHookVDotNet.ini' and 'ScriptHookVDotNet.asi' ("ScriptHookVDotNet.zip" >> Unzip) from https://github.com/crosire/scripthookvdotnet/releases 
3. Download 'NativeUI.dll' ("Release.zip" >> Unzip) from https://github.com/Guad/NativeUI/releases

4. Move the following items to the main GTA V game folder: 
   - 'ScriptHookV.dll', 'dinput8.dll', 
   - 'ScriptHookVDotNet2.dll', 'ScriptHookVDotNet.ini', 'ScriptHookVDotNet.asi', 
   - 'GTAVisionUtil.asi' (included) and 'GTAVisionUtil.lib' (included)
 
5. Clone drs/gta. Move 'DRS.dll' (included), 'NativeUI.dll',  to a GTA V game subfolder.

6. Change 

a. If using Steam: Disable the steam overlay for this game.
b. Settings:
	- Display: 
		Radar: Off
		HUD: Off
	- Graphics:
		Screen Type: Windowed Borderless
		Resolution: 1920 x 1080
		MSAA: Off
		Distance Scaling: None
	- Advanced Graphics:
		Extended Distance Scaling: None

##### In-game

1. Press F10 to show the DRS Menu
2. Select 'Experiment' >> 'Wide-view only'
3. Enter directory for raw output: 'XXXXXXW00RD.json', 'XXXXXXW00RD_colour.tif', 'XXXXXXW00RD_depth.tif', 'XXXXXXW00RD_stencil.tif'
4. Enter the number of desired runs

 

		
 

