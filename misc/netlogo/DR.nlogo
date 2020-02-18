; ============================================= ;
;                    GLOBAL                     ;
; ============================================= ;

globals [
  drone_base               ; co-ordinates of drone base
  drone_speed              ; drone speed

  total_respond_1          ; total number of low-level accidents responded to
  total_respond_2          ; total number of medium-level accidents responded to
  total_respond_3          ; total number of high-level accidents responded to

  total_not_respond_1      ; total number of low-level accidents not responded to
  total_not_respond_2      ; total number of medium-level accidents not responded to
  total_not_respond_3      ; total number of high-level accidents not responded to

  maintenance_count        ; total count of the number of drone returns to the drone base

  xy_roads                 ; road agentset
  xy_buildings             ; buldings agentset
  xy_obstructions          ; road agent subset containing obstructions
]

; ============================================= ;
;                     BREEDS                    ;
; ============================================= ;

breed [accidents accident]
accidents-own [
  time_remaining       ; waiting time remaining
  emergency_level      ; low, medium, high
]

breed [drones drone]
drones-own [
  state                ; available or unavailable
  energy               ; flight time remaining
]

patches-own[           ; height of building
  height
]

; ============================================ ;
;                    SETUP                     ;
; ============================================ ;

to setup
  clear-all
  setup-patches
  setup-accidents
  setup-drones
  reset-ticks
end

; patches

to setup-patches

  ; ============ building ===========

  ; building co-ordinates

  let Grid 3   ; results in 2 patches between every building
  set xy_buildings patches with [
    pxcor mod Grid = 0 and
    pycor mod Grid = 0
  ]

  ; building heights

  ask xy_buildings [
    set height (1 + round ((random (building-height + 1)) / 10 * 8))
    set pcolor height
  ]

  ; =========== drone base ===========

  ; drone base location

  ask min-one-of xy_buildings [
    distancexy x-drone-base y-drone-base
  ][
    set pcolor blue
  ]

  set drone_base patches with [
    pcolor = blue
  ]

  ; =========== obstruction ===========

  ; obstruction co-ordinates

  set xy_obstructions n-of num-obstructions patches with [
    not member? self xy_buildings and
    not member? self patch-set [neighbors] of drone_base
  ]

  ask xy_obstructions [
    set pcolor white
  ]

  ; ============== road ===============

  ; road co-ordinates

  set xy_roads patches with [
    not member? self xy_buildings and
    not member? self xy_obstructions
  ]

end

to setup-accidents
  set-default-shape accidents "x"
end

to setup-drones
  set drone_speed (1 - 0.05 * conditions)
  set-default-shape drones "airplane"
  create-drones num-drones [

    ; drone features

    set state "available"
    set color green          ; drone status = available

    set energy battery-capacity

    ; locate drones about drone base

    move-to one-of drone_base
    let candidates neighbors with [
      not any? drones-here
    ]
    if any? candidates [
      move-to one-of candidates
    ]
  ]

end

; ============================================ ;
;                      GO                      ;
; ============================================ ;

to go
  cause-accident
  move-drones
  countdown-accident
  if not any? drones [
    stop
  ]
  tick
end

; ================= ACCIDENTS ================= ;

to-report accident-severity-distribution
  ; accident severity levels

  let low 1
  let medium 2
  let high 3

  ; accident severity instance

  let severity random-exponential 1
  ifelse severity <= 1 [
    report low
  ][
    ifelse severity <= 2 [
      report medium
    ][
      report high
    ]
  ]
end

to cause-accident

  ; if accident occurs within a tick

  let occurence random-poisson accident-frequency

  ; features of accident at time of occurence

  if occurence > 0 [
    create-accidents 1 [
      set emergency_level accident-severity-distribution         ; low, medium, high
      ifelse emergency_level = 1 [
        set color green
        set time_remaining low-severity
      ][
        ifelse emergency_level = 2 [
          set color orange
          set time_remaining mid-severity
        ][
          set color red
          set time_remaining high-severity
        ]
      ]
      move-to one-of xy_roads
    ]
  ]
end

to countdown-accident

  ; countdown active accidents

  ask accidents with [
    time_remaining > 0
  ][
    set time_remaining (time_remaining - 1)
  ]

  ; remove accidents if time remaining is less than 0 AND keep count

  ask accidents with [
    time_remaining <= 0
  ][
    ifelse emergency_level = 1 [
      set total_not_respond_1 (total_not_respond_1 + 1)
    ][
      ifelse emergency_level = 2 [
        set total_not_respond_2 (total_not_respond_2 + 1)
      ][
        set total_not_respond_3 (total_not_respond_3 + 1)
      ]
    ]
    die
  ]
end

; =================== DRONES ================= ;

; Movement functions

to-report moveable-patch?
  let answer False
  if (pcolor = "black") or                   ; road
  (height < drone-flight-height) and not     ; building height less than drone
  (pcolor = white) [                         ; obstruction
    set answer True
  ]
  report answer
end

to step-drone
  fd drone_speed                             ; drone speed
  set energy (energy - discharge-rate)       ; energy discharge
  if energy <= 0 [                           ; drone death condition
    die
  ]
end

to-report patch-here-drone-base
  let answer False
  let drone_neighbors patch-set [neighbors] of drone_base
  if member? patch-here drone_neighbors [
   set answer True
  ]
  report answer
end

to state-drone

  ; NOT at drone base

  ifelse not patch-here-drone-base [
    if energy <= refuel-energy-level [

      ; state = returning to recharge

      set state "unavailable"
      set color pink
    ]
  ][

    ; at drone base

    ifelse energy < battery-capacity [
      if state = "unavailable" [

        ; state = started recharging

        set state "recharging"
        set maintenance_count (maintenance_count + 1)
      ]
      if energy > battery-capacity / 2 [

        ; state = more than half battery capacity whilst recharging

        set color orange
      ]
    ][
      if state = "recharging" [

        ; state = fully recharged

        set energy battery-capacity
        set state "available"
        set color green
      ]
    ]
   ]
end


to move-drones

  ask drones [

    ; determine drone state

    state-drone

    ifelse state = "recharging" [

      ; state = recharging

      set energy (energy + charge-rate)
    ][

      ; state = available or unavailable

      let candidate_patches neighbors with [moveable-patch?]

      ifelse state = "unavailable" [

        ; state = unavailable
        ; find next patch path closest to the drone base

        let return_patch min-one-of candidate_patches [
          distancexy x-drone-base y-drone-base
        ]

        ; move to candidate patch that is closest to the drone base

        set heading towards return_patch
        step-drone
      ][

        ; state = available
        ; determine if accident on candidate patches (ordered by severity)

        ifelse any? accidents-on candidate_patches [
          let chosen_accident one-of accidents-on candidate_patches
          let candidate_accidents accidents-on candidate_patches
          ask candidate_accidents [
            ifelse (count candidate_accidents with [emergency_level = 3]) > 0 [
              set chosen_accident one-of candidate_accidents with [emergency_level = 3]
            ][ifelse (count candidate_accidents with [emergency_level = 2]) > 0 [
                set chosen_accident one-of candidate_accidents with [emergency_level = 2]
             ][
                set chosen_accident one-of candidate_accidents with [emergency_level = 1]
             ]
            ]
          ]

          ; move to accident patch

          set heading towards chosen_accident
          step-drone

          ; remove accident

          let accident_patch [patch-here] of chosen_accident

          if any? drones-on accident_patch [
            ask accidents-on accident_patch [
              ifelse emergency_level = 1 [
                set total_respond_1 (total_respond_1 + 1)
              ][
                ifelse emergency_level = 2 [
                  set total_respond_2 (total_respond_2 + 1)
                ][
                  set total_respond_3 (total_respond_3 + 1)
                ]
              ]
              die
            ]
          ]
        ][

          ; state = available
          ; move to random candidate patch

          set heading towards one-of candidate_patches
          step-drone
        ]
      ]
    ]
  ]

end
@#$#@#$#@
GRAPHICS-WINDOW
332
20
876
565
-1
-1
13.0732
1
10
1
1
1
0
0
0
1
-20
20
-20
20
1
1
1
ticks
30.0

SLIDER
168
164
315
197
accident-frequency
accident-frequency
0.01
0.1
0.05
0.01
1
NIL
HORIZONTAL

SLIDER
169
316
320
349
num-drones
num-drones
1
9
9.0
1
1
NIL
HORIZONTAL

BUTTON
99
525
163
558
NIL
Setup
NIL
1
T
OBSERVER
NIL
S
NIL
NIL
1

BUTTON
169
525
232
558
NIL
Go
T
1
T
OBSERVER
NIL
G
NIL
NIL
0

TEXTBOX
9
10
63
30
World
16
0.0
0

TEXTBOX
9
222
70
242
Drones
16
0.0
1

SLIDER
170
232
316
265
x-drone-base
x-drone-base
-18
18
0.0
3
1
NIL
HORIZONTAL

SLIDER
170
37
314
70
building-height
building-height
1
10
5.0
1
1
NIL
HORIZONTAL

SLIDER
170
274
318
307
y-drone-base
y-drone-base
-18
18
0.0
3
1
NIL
HORIZONTAL

SLIDER
170
78
314
111
conditions
conditions
1
10
2.0
1
1
NIL
HORIZONTAL

SLIDER
169
435
306
468
charge-rate
charge-rate
0
1
1.0
0.05
1
NIL
HORIZONTAL

SLIDER
169
474
307
507
discharge-rate
discharge-rate
0
1
1.0
0.05
1
NIL
HORIZONTAL

SLIDER
170
360
322
393
drone-flight-height
drone-flight-height
1
10
5.0
1
1
NIL
HORIZONTAL

SLIDER
169
121
314
154
num-obstructions
num-obstructions
0
20
3.0
1
1
NIL
HORIZONTAL

PLOT
893
48
1053
198
Awaiting Response
Ticks
% of Accidents
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"Low" 1.0 0 -10899396 true "" "plot (count accidents with [emergency_level = 1]) / count accidents"
"Medium" 1.0 0 -955883 true "" "plot (count accidents with [emergency_level = 2]) / count accidents"
"High" 1.0 0 -2674135 true "" "plot (count accidents with [emergency_level = 3]) / count accidents"

PLOT
1058
48
1272
198
Responded
Ticks
% Responded
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"Low" 1.0 0 -10899396 true "" "plot total_respond_1 / total_not_respond_1"
"Mid" 1.0 0 -955883 true "" "plot total_respond_2 / total_not_respond_2"
"High" 1.0 0 -2674135 true "" "plot total_respond_3 / total_not_respond_3"

TEXTBOX
32
45
182
63
Mean building height
11
105.0
1

TEXTBOX
31
75
181
117
Weather conditions \n(affects drone speed)\n
11
105.0
1

TEXTBOX
33
120
183
148
Number of obstructions \n(drone no-fly zones)
11
105.0
1

TEXTBOX
32
169
182
197
Mean accident occurrence\nper tick
11
105.0
1

TEXTBOX
30
261
180
279
Drone base co-ordinates
11
15.0
1

TEXTBOX
30
327
180
345
Number of drones
11
15.0
1

TEXTBOX
30
373
180
391
Drone flight height
11
15.0
1

TEXTBOX
113
412
231
430
Drone battery capacity
11
15.0
1

SLIDER
24
474
162
507
refuel-energy-level
refuel-energy-level
10
40
20.0
1
1
NIL
HORIZONTAL

SLIDER
24
436
161
469
battery-capacity
battery-capacity
10
100
50.0
1
1
NIL
HORIZONTAL

MONITOR
889
409
979
454
% - Responded
((total_respond_1 + total_respond_2 + total_respond_3)/(total_not_respond_1 + total_not_respond_2 + total_not_respond_3 + total_respond_1 + total_respond_2 + total_respond_3)) * 100
0
1
11

MONITOR
891
518
980
563
X-to-Drones 
count accidents / count drones
2
1
11

TEXTBOX
890
17
962
37
Accidents
16
0.0
1

TEXTBOX
888
292
945
312
Drones
16
0.0
1

MONITOR
890
464
979
509
% - Battery
((round ((sum [energy] of drones) / (count drones)))/ battery-capacity) * 100
0
1
11

PLOT
1072
412
1272
562
Drone State
Ticks
% - Drones
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"Unavailable" 1.0 0 -2064490 true "" "plot ((count drones with [state = \"unavailable\"]) / (count drones)) * 100"
"Recharging" 1.0 0 -955883 true "" "plot ((count drones with [state = \"recharging\" ]) / (count drones)) * 100"
"Available" 1.0 0 -10899396 true "" "plot ((count drones with [state = \"available\"]) / (count drones)) * 100"

INPUTBOX
1296
339
1346
399
X-high
1000.0
1
0
Number

INPUTBOX
1348
339
1398
399
X-mid
500.0
1
0
Number

INPUTBOX
1400
339
1450
399
X-low
250.0
1
0
Number

MONITOR
1346
519
1452
564
Value (R)
X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count
2
1
11

TEXTBOX
1357
283
1400
303
Profit
16
0.0
1

SLIDER
1307
70
1427
103
low-severity
low-severity
10
500
200.0
10
1
NIL
HORIZONTAL

SLIDER
1308
117
1425
150
mid-severity
mid-severity
10
300
100.0
10
1
NIL
HORIZONTAL

SLIDER
1307
163
1425
196
high-severity
high-severity
10
100
50.0
10
1
NIL
HORIZONTAL

BUTTON
1071
326
1186
359
watch random drone
watch one-of drones
NIL
1
T
OBSERVER
NIL
W
NIL
NIL
0

BUTTON
1071
370
1188
403
reset perspective
rp
NIL
1
T
OBSERVER
NIL
R
NIL
NIL
0

MONITOR
889
356
977
401
% - Alive
(count drones / num-drones) * 100
0
1
11

TEXTBOX
1311
16
1426
35
Additional Settings:
12
105.0
1

TEXTBOX
1295
36
1444
54
Maximum accident waiting time
11
15.0
1

TEXTBOX
947
225
1130
253
% average accident time remaining
11
15.0
1

TEXTBOX
896
224
941
242
Output:
12
105.0
1

MONITOR
1125
208
1184
253
% - Low
(((sum [time_remaining] of accidents with [emergency_level = 1]) / (count accidents with [emergency_level = 1])) / low-severity) * 100
0
1
11

MONITOR
1185
208
1242
253
% - Mid
(((sum [time_remaining] of accidents with [emergency_level = 2]) / (count accidents with [emergency_level = 2])) / mid-severity) * 100
0
1
11

MONITOR
1247
208
1309
253
% - High
(((sum [time_remaining] of accidents with [emergency_level = 3]) / (count accidents with [emergency_level = 3])) / high-severity) * 100
0
1
11

TEXTBOX
889
327
936
345
Output: 
12
105.0
1

TEXTBOX
940
328
1027
346
CONTROL PANEL
11
15.0
1

TEXTBOX
986
359
1047
401
% of initial drones operating
11
15.0
1

TEXTBOX
988
410
1059
452
% of accidents responded to
11
15.0
1

TEXTBOX
988
473
1062
501
% of average drone battery
11
15.0
1

TEXTBOX
990
520
1059
559
Ratio of accidents to drones
11
15.0
1

TEXTBOX
1297
318
1365
336
Revenue: 
12
105.0
1

TEXTBOX
1354
318
1410
348
Accidents
12
15.0
1

TEXTBOX
1300
412
1332
430
Cost:
12
105.0
1

TEXTBOX
1333
412
1376
430
Drones
12
15.0
1

INPUTBOX
1298
435
1368
495
Initial
75000.0
1
0
Number

INPUTBOX
1371
435
1451
495
Maintenance
100.0
1
0
Number

TEXTBOX
1302
520
1339
538
Profit:
12
105.0
1

@#$#@#$#@
## WHAT IS IT?

The DRONE RESPONSE SERVICE (DRS) model simulates the operation of autonomous emergency response drones within an urban environment. The purpose of this simulation is to understand the effect of both environmental and drone-build constraints on the ability of a specified number of drones to respond to unforeseen accident events. This analysis is intended to offer an initial due diligience test for a business which may plan to offer a DRS service. 

## HOW IT WORKS AND HOW TO USE IT

The model consists 3 agent categories:

### PATCH-RELATED: Buildings, obstructions and roads

The urban environment is represented by a fixed 21 x 21 patch grid.  

#### a. Buildings

Buildings are equally spaced within this grid environment. Each building is: 

 - Placed 2 patches a part from the next building and/or environment border.
 
- Assigned a uniform random height between 1 and a user-influenced "building-height" variable. The maximum height range allowed is 1 - 9. This height property is denoted by shading each building's patch in proportion to its height. The taller the building, the lighter the patch colour.

A Drone Base (DB) is chosen from the set of buildings. This base is the building that is closest to a set of co-ordinates ("x-drone-base", "y-drone-base") input by the user. The colour of this building is set to blue.

#### b. Obstructions

The user may specify a certain number of obstructions within the urban environment using the "num-obstructions" variable. An obstruction is a patch over which a drone is not permitted to pass. They are randomly located at patches that are both (i) not buildings and (ii) not a neighbour of the DB. An obstruction patch is given a white colour. This alludes to the equivalent height of the patch (i.e. white is the lightest possible shade).

#### c. Roads

Roads are a subset of patches that are neither buildings nor obstructions. They are shaded black to represent their lack of height (i.e. they are located at ground level). 
    
### DRONES

An "airplane" symbol is used to designate a drone. The user defines the number of drones to be included in the model with the "num-drones" slider. At any given time, a drone may be in 1 of 3 states (and will be coloured accordingly), namely: (i) "available" (green) (ii) "unavailable" (pink) (iii) "recharging". A drone's location and energy levels will determine its current state. 

A synopsis of a drone's life will be described below:

#### a. Initialisation

All drones are: 

- Assigned their maximium energy capacity. This capacity level is defined by the user through the "battery-capacity" variable.

- Given a movement speed. This speed is a value from 0.5 - 0.95, and represents the rate at which the drone moves forward per tick. The drone speed is assumed to be affected by the prevailing weather conditions. The user is thus able to influence the drone speed by adjusting the "conditions" variable. A lower "weather condition" value indicates better weather conditions, and vice versa.

- Located on a unique patch about (i.e. neighbouring patches) or on the DB. 

- In an "available" state and coloured green accordingly.

#### b. Movement

A drone may only move to a patch if and only if:

- the building on said patch has a height less than that of the user-defined "drone-flight-height" or

- the patch does not contain an obstruction or

- the patch represents a patch of road 

A drone's visual range encompasses all immediate neighbouring patches. If an accident is present on a neighbouring patch, the drone will face and move towards the accident patch. This will cause the accident to disappear. Multiple accidents of different severities located on the drone's neighbouring patches will cause the drone to prioritise more severe accidents. 
 
Movement involves the drone expending energy. As such, the drone's energy level will decrease by the user-defined "discharge-rate" upon each movement. If the drone's energy level falls to or below 0 after a movement, the drone will die/crash.

#### c. Recharging energy

When the drone's energy drops below the user-defined "refuel-energy-level", the drone will seek to return to the DB to recharge. The state of the drone will become "unavailable" [colour turns pick] and its flight rules will change. The drone willl fly to allowable patches with the closest distance to the DB. 

Upon reaching an immediate patch neighbour of the DB, the drone will recharge at the user-specified "charge-rate". A maintenance cost is charged upon each drone return to the DB. This cost may be assigned by the user under the "Profit" section. The state of the drone will then change from "unavailable" to "recharging". The colour of the drone will change from pink to orange when the energy level of the drone exceeds half its battery capacity. The drone will recharge fully before becoming "available" again.   

### ACCIDENTS

#### a. Accident occurrence

An accident is denoted by an "X" symbol. 1 accident may occur per tick. This occurrence is dependent on a random value from a Poisson Distribution with parameter lambda = "accident-frequency", as specified by the user. A larger "accident-frequency" value will increase the chance that an accident may occur. 

#### b. Accident severity

If an accident does occur, the accident may occur with 3 different levels of severity: (i) Low (ii) Mid (iii) High. The severity of the accident is determined by a random value from an Exponential Distribution with parameter lambda = 1. This ensures more severe accidents happen less frequently. Low-, mid-, high-severity accidents are denoted by green, orange and red colours respectively. The severity of an accident defines the "time-remaining": The maximum amount of time to which response is deemed valuable. If an accident is not responded to within this period of time, it will lapse/disappear. This will count against the DRS's response rate. The "time-remaining" variable may be set for each severity level by the user under the "Additional Settings" section. The longer the "time-remaining", the greater the chance that the drones may respond to the accident.    

#### c. Accident location

If an accident does occur, it will be placed on a random road or obstruction patch throughout the environment. The permissibility of an accident landing on an obstruction patch attempts to emulate situations in which a drone is unable to respond to all accidents.

#### d. Accident revenue

The DRS is envisioned to be a viable business. Its customers (e.g. insurers, police, emergency response services) will therefore be charged per accident, and according to the severity of the accident, as responded to by the drones. More severe accidents are considered to require a greater drone "hang time" in practice, and thus may be charged more for the service. The user may specify the revenue he/she expects to receive from each accident type under the "Profits" section.  

## DISPLAY CONSOLE AND THINGS TO NOTICE (TTN)

The main things to notice are:

- The drones tend to explore the city within an expanding radius about the drone base, before heading back to the drone base to recharge. 

- At initialisation, the drones start flying at the same time. This causes them to return to the drone base at similar times as well. However, this clustering effect tends to reduce over time. 

### ACCIDENTS

#### a. Graph: Awaiting response

Percentage of severity-specific accidents awaiting response out of all awaiting accidents.  

TTN: The percentage for severity-specific accidents is proportional to one another. As low-severity accidents occur more frequently than mid- and high-severity accidents, they usually account for the greatest percentage of accidents awaiting response. 

#### b. Graph: Responded

Percentage of severity-specific accidents responded to out of all severity-specific accidents that have occurred.

TTN: As observed for the "awaiting response graph", the percentages are proportional to one another. One would expect to see the percentage increase as the severity decreases, due to the longer waiting times associated with lower severity levels.

#### c. Monitor: % average accident time remaining

The average viable response time remaining of accidents with a particular severity level.

TTN: It changes (up and down) as accidents occur and disappear.

### DRONES

#### a. Monitor: % of initial drones operating

Percentage of the initially selected number of drones that are still operational (i.e. have an energy value > 0). 

TTN: All drones, with the default settings, should remain operational. Creating a poor environment and drone-battery settings (refer things to try) may cause the drones to die. If this does occur, the drones tend to die one-by-one.  

#### b. Monitor: % of accidents responded to

Percentage of all accidents responded to, relative to the total number of accidents that have occurred. 

TTN: Almost all accidents are responded to initially. As time ticks over, the average number of accidents "missed" increases. Accidents occurring outside the radius reach of the drones, or those occurring during their recharge times tend to be missed. The % of accidents responded to therefore decreases rapidly during drone recharging times. The monitor remains fairly stable or increases when the drones are in an "available" state. This percentage should plateau at some low level over time.      

#### c. Monitor: % of average drone battery

Percentage of the average battery capacity remaining, out of all operational drones.

TTN: As alluded to above, the drones leave and return to the drone base at similar times. Therefore, the drones' average battery capacity tend to decrease from the user-defined battery capacity in proportion to the discharge rate (if any), when the drones are "available". If the minimum refuel energy level is reached, the drones will attempt to return to base. The average battery capacity will continue to decrease until they are at the drone base. Their average battery level will increase towards their battery capacity. The drones become "available" once more and the process is then repeated.
 
#### d. Monitor: Ratio of accidents to drones

Ratio of accidents awaiting response to the total number of operational drones.

TTN: Ratio decreases and increases as accidents occur, are responded to and disappear. 

#### e. Graph: Drone state

Percentage of operational drones that are either in a/an "available", "unavailable" or "recharging" state. 

TTN: As the drones recharging times tend to be similar, all the drones are usually in the same state. This results in 1 state dominating the graph at a time. 

### PROFIT

To be a viable business project, the DRS will need to be profitable. Its novel nature makes it difficult to forecast the revenue to be received or costs to be incurred whilst operating the service. As such, the user is allowed to input values that correspond to aggregate estimates for these quantities.   

#### a. Revenue: Accidents

Responding to accidents of greater severity are assumed to earn the operator greater revenues. Prudent default revenue assumptions are made to reflect the level of uncertainty in the actual revenues received. 

#### b. Cost: Drones

Drone costs are divided into 2 main categories: initial and maintenance costs. Initial costs refer to the cost of purchasing and modifying the drone for its desired purpose. Commercial drones, depending on its features, may cost in the region of R 50 000 - R 100 000 (4/9/2018). If the business were to be expanded over time, the initial drone costs would be reduced as a result of economies of scale. The default initial drone cost is estimated at R 75 000. Maintenance costs represent ongoing costs of maintaining and recharging the drones. The high level of uncertainty in the aggregate level of these costs also necessitates a prudently assumed value (i.e. R 100 per trip to the drone base, per drone).  

#### c. Profit

The profit value is calculated by subtracting costs from revenues.

TTN: The profit value is negative initially to account for the high initial costs of purchasing the drones. Over time, the profit value increases or decreases slowly depending on the users' parameter choices. Whereas responding to accidents increases revenues, returning to base decreases them due to maintenance costs. A drone "outing" may therefore be profitable or unprofitable depending on whether an accident was responded to.

## THINGS TO TRY

### MEASURES

The model has 3 main measures on which to base DRS performance, namely:

- Percentage of drones remaining (PD)

- Percentage of accidents responded to (PR)

- Relative changes to profit (PFT)

The user should attempt to modify parameters to maximise and minimise one/all of the aforementioned measures. 

### PARAMETERS

Extreme and optimal ENVIRONMENT, DRONE and ACCIDENT settings should be set to understand their relative importance. An example of extreme environment conditions is provided below. The converse settings will produce an optimal environmental configurations. 

To configure the harshest environment, set:

- "building-height" to 10: drones will have to go around buildings that are taller than its flight height.

- "conditions" to 10: drone speed will be reduced dramatically.  

- "num-obstructions" to 20: drone movement is more constrained. 

The drones' overall reach will contract, decreasing the effectiveness of the DRS in maximising PR.   

### FUN THINGS TO TRY

#### a. Drone-related

- Kill the drones as quickly as possible ["refuel-energy-level" = 10, "conditions" = 10, "discharge-rate" = 1].

- Watch a random drone [via button].

- Try prevent the drones from leaving the drone base ["refuel-energy-level" = 90].

- Let the DRS be under-staffed. See the impact where "num-drones" = 1. 

- Assess the benefit of a "no recharge" model ["discharge-rate" = 0].

#### b. Accident-related

- Cause a DRS nightmare. Create accidents everywhere [maximise "accident-frequency"]. 

- Let accidents be "patient" or "impatient". Change the accidents' maximum response times to long or short.

#### c. Environment-related

- Create the "ice rink" effect. Let the environment be flat ["building-height" = 1, "num-obstuctions" = 0] and allow the drones to fly at maximum speed ["condition" = 1].

- Move the drone base to a "corner" location ["x-drone-base" = -12, "y-drone-base = -12]

## EXTENDING THE MODEL

The drones were chosen to act independently of each other, in order to facilitate their modelling as agents. An improvement to the model may decrease the granularity of the agents, instead choosing to model the DRS as an agent. This would allow:

- Accident locations to be known (if provided to a central system). The system would then allocate the drone that is best positioned to respond. An accident waiting list may be formed. The next available drone may then respond to a selected accident based on its position on the waiting list, distance from the drone, time remaining and its severity. OR 

- Accident locations to be discovered (if not provided to a central system). The drones may be able to communicate identified accident locations with one another. These may include locations it is both responding, and not responding, to. OR

- Accident locations to be known and discovered.

- High-accident zones to be identified. The system could then pre-empt the occurrence of accidents, and direct more drones to these areas when they are not busy. Accidents would then need to be remodelled to occur in certain areas with a greater probability.

- Different kinds of drones to be deployed. Each may have different features suited for a particular purpose. Fast flying drones may act to discover new accidents or respond to severe accidents, whereas long battery life drones may respond to less severe accidents.

Further improvements may include:

- Having multiple drone base locations. This would enable the drones to recharge at different locations, increasing their reach and "available" time.

- Using an environment map of a real city. This would better represent the operating space of the drones.

- Allowing for more detail in weather conditions (e.g. wind, rain). The weather may also dynamically change over time.

- Building an equivalent 3D model. 

- Incorporating more empirical evidence into the parameterisation and scaling of all the models' features. 

## RELATED MODELS

The Ants model incorporates a similar random movement technique for the ants as employed by the drones.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.0.4
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="Factor 1: Accident frequency" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <steppedValueSet variable="accident-frequency" first="0.01" step="0.01" last="0.1"/>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Factor 2: Conditions" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <steppedValueSet variable="conditions" first="1" step="1" last="10"/>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Factor 3: Drone Base" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <steppedValueSet variable="y-drone-base" first="-12" step="12" last="12"/>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <steppedValueSet variable="x-drone-base" first="-12" step="12" last="12"/>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Factor 5: Refuel energy level" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <steppedValueSet variable="refuel-energy-level" first="10" step="5" last="100"/>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Factor 4: Battery capacity" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <steppedValueSet variable="charge-rate" first="0.5" step="0.5" last="1"/>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <steppedValueSet variable="battery-capacity" first="50" step="50" last="100"/>
    <enumeratedValueSet variable="conditions">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <steppedValueSet variable="discharge-rate" first="0.5" step="0.5" last="1"/>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Factor 6: Number of drones" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="5"/>
    </enumeratedValueSet>
    <steppedValueSet variable="num-drones" first="1" step="1" last="9"/>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Scenario 1: Optimistic" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="15"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.07"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="-12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Scenario 2: Best-estimate" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="20"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.05"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="-12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="Scenario 3: Prudent" repetitions="10" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="30000"/>
    <metric>round (count drones / num-drones) * 100</metric>
    <metric>round (((total_respond_1 + total_respond_2 + total_respond_3) / (total_respond_1 + total_respond_2 + total_respond_3 + total_not_respond_1 + total_not_respond_2 + total_not_respond_3)) * 100)</metric>
    <metric>X-high * total_respond_1 + X-mid * total_respond_2 + X-low * total_respond_3 - Initial * num-drones - Maintenance * maintenance_count</metric>
    <enumeratedValueSet variable="building-height">
      <value value="7"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="low-severity">
      <value value="200"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="charge-rate">
      <value value="0.5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="drone-flight-height">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-low">
      <value value="250"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="refuel-energy-level">
      <value value="25"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="battery-capacity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="conditions">
      <value value="2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-drones">
      <value value="9"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="high-severity">
      <value value="50"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-mid">
      <value value="500"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Maintenance">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="y-drone-base">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="num-obstructions">
      <value value="3"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="discharge-rate">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="accident-frequency">
      <value value="0.03"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="Initial">
      <value value="75000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="x-drone-base">
      <value value="-12"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="mid-severity">
      <value value="100"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="X-high">
      <value value="1000"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
