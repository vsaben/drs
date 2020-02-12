## Overview

This repository provides the coding scripts used in fulfillment of the Master's thesis titled: "Car accident feature extraction from a drone-based video feed". 

### Thesis overview 

The thesis introduces a novel drone-based infrastructure for responding to and recording urban car accidents (UCAs), known as a Drone Response Service (DRS). Upon the occurrrence of a car accident, a drone:

1. Autonomously navigates to the accident site.
2. Records visual evidence of the damaged vehicles or property.
3. Relays this information to insurance providers and other interested parties.   

The focus is to develop the computer vision tools to support this operation. The localisation and classification of UCA damage is broken down into 3 stages. These stages inform this thesis's objectives, namely:

1. Identification of at least 1 vehicle in an image frame.
2. Localisation of all damaged and non-damaged vehicles.  
3. Localisation and classification of vehicle damage.  

### Machine Learning

YOLO-based networks are used to generate region proposals for a multi-task network head. The head encompasses 5 tasks, computed in parallel: 

1. Classification: Damage 
2. Classification: Car class
3. Object Detection: 3D Bounding Box (BB) co-ordinates
4. Instance Segmentation: Car
5. Instance Segmentation: Damage

#### Backend models evaluated:

1. YOLOv3
    - Normal
    - Slim
2. YOLOv3-tiny
    - Normal
    - FPN

### Data

Car accidents are simulated in Grand Theft Auto (GTA) V.   

