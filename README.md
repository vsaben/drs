This repository supports the autonomous extraction and dense-annotation of simulated vehicle accident scenarios, as viewed from an aerial perspective, from Grand Theft Auto V (GTA). The data are used to train a modified, YOLO-based 2-stage: 

1. Object detector for far-view vehicle localisation, pose estimation and damage status classification.  
2. Instance segmentation model for the close-view identification of specified vehicle damage types.

Aerial vehicle and vehicle damage (VAVD) underly the operation of a novel drone-based infrastructure for responding to and recording urban vehicle accidents (UVAs), hereon referred to as a Drone Response Service (DRS). All coding scripts are provided as an addendum to the thesis titled: "Vehicle accident feature extraction from a drone-based video feed". 

### Thesis overview 

The thesis introduces a novel , known as a Drone Response Service (DRS). Upon the occurrrence of a car accident, a drone:

1. Autonomously navigates to the accident site.
2. Records visual evidence of the damaged vehicles or property.
3. Relays this information to insurance providers and other interested parties.   

The focus is to develop the computer vision tools to support this operation. The localisation and classification of UCA damage is broken down into 3 stages. These stages inform this thesis's objectives, namely:


### Machine Learning

YOLO-based networks are used to generate region proposals for a multi-task network head. The head encompasses 5 tasks, computed in parallel: 
