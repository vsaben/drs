using System;
using System.Collections.Generic;
using System.Linq;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public static class Accident
    {

        // 1: Methods ==============================================================================================

        /// A: Overall: Cause accident

        public static Target Cause(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {                                         
            //// i: Choose AND cause accident

            AccidentType accidenttype = Choice(runcontrol, testcontrol);                       // [a] Choose accident

            testcontrol.accidentinitiatetime = World.CurrentDayTime;                           // [b] Update accident initiation time
            Target target = Accident.AccidentFunctions[accidenttype](runcontrol, environment); // [c] Cause accident

            //// ii: Wait for accident to occur

            testcontrol.timeofaccident = World.CurrentDayTime;                                 // [a] Update time of accident
            Transition.SetPersistence(target, true);                                           // [b] Maintain persistence

            return target;
        }

        /// B: Choose accident

        public enum AccidentType
        {
            RanTrafficControl,             // 1: Ran traffic control
            RearEndCollision,              // 2: Rear end collision

            // 3: Ran off road

            DriverDies,
            ReduceGrip,
            ExplodeTyre,
            SteerOff

            // LaneChange, 
            // RightTurnOncoming
            // Speed 
        }

        public static AccidentType Choice(RunControl runcontrol, TestControl testcontrol)
        {
            Array accidentvalues = Enum.GetValues(typeof(AccidentType));
            AccidentType accidenttype = (AccidentType)accidentvalues.GetValue(runcontrol.random.Next(accidentvalues.Length));
            testcontrol.incident = accidenttype.ToString();
            return accidenttype;
        }

        public static IDictionary<AccidentType, Func<RunControl, Environment, Target>> AccidentFunctions =
            new Dictionary<AccidentType, Func<RunControl, Environment, Target>>{
                { AccidentType.RanTrafficControl, AccidentMethods.RanTrafficControl},    // 1: Ran traffic control
                { AccidentType.RearEndCollision, AccidentMethods.RearEndCollision},      // 2: Rear end collision

                // 3: Ran off road

                { AccidentType.DriverDies, AccidentMethods.DriverDies},
                { AccidentType.ReduceGrip, AccidentMethods.ReduceGrip},
                { AccidentType.ExplodeTyre, AccidentMethods.ExplodeTyre},
                { AccidentType.SteerOff, AccidentMethods.SteerOff}

                //{ AccidentType.LaneChange, AccidentMethods.LaneChange},
                //{ AccidentType.RightTurnOncoming, AccidentMethods.RightTurnOncoming},
                //{ AccidentType.Speed, AccidentMethods.Speed}
            };
    }

    public static class AccidentMethods
    {
        // Paper A: Classifying urban car crashes for countermeasure development

        // TYPE 1: Ran traffic control =======================================================================================================================

        public static Target RanTrafficControl(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                 // [a] Select random moving vehicl
            Target target = Target.Setup(runcontrol, res_vehicle);                                                        // [b] Initialise target

            Ped ped = Function.Call<Ped>(Hash.GET_PED_IN_VEHICLE_SEAT, target.vehicle, -1);                               // [c] Select driver ped

            float maxspeed = Function.Call<float>(Hash._GET_VEHICLE_MAX_SPEED, target.vehicle);                           // [d] Increase speed
            float currentspeed = target.vehicle.Speed;
            float midmaxspeed = currentspeed + (maxspeed - currentspeed) / 2;
            float newspeed = midmaxspeed + (float)(((maxspeed - currentspeed) / 2) * runcontrol.random.NextDouble());

            Function.Call(Hash.TASK_VEHICLE_DRIVE_WANDER, ped, target.vehicle, newspeed, (int)DrivingStyle.IgnoreLights); // [e] Allow reckless driver to wander

            return target;
        }

        // TYPE 2: Stopped/stopping ==========================================================================================================================

        public static Target RearEndCollision(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                 // [a] Select random moving vehicle
            Target target = Target.Setup(runcontrol, res_vehicle);                                                        // [b] Initialise target

            IList<Vehicle> ClosestActiveVehicles = target.nearbyvehicles.Where(o => o.IsAlive).ToList();                  // [c] Determine the closest active vehicle
            Vehicle closestactivevehicle = target.nearbyvehicles.OrderBy(o => World.GetDistance(target.location, o.Position)).ToList()[0];

            Ped ped = Function.Call<Ped>(Hash.GET_PED_IN_VEHICLE_SEAT, closestactivevehicle, -1);                         // [d] Select driver ped

            Function.Call(Hash.TASK_VEHICLE_CHASE, ped, target.vehicle);                                                  // [e] Set driver ped to chase the target vehicle
            Function.Call(Hash.SET_TASK_VEHICLE_CHASE_IDEAL_PURSUIT_DISTANCE, ped, 0);                                    // [f] Set pursuit distance to 0

            return target;
        }

        // TYPE 3: Ran off road ==============================================================================================================================

        public static Target DriverDies(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                 // [a] Select random moving vehicle
            Target target = Target.Setup(runcontrol, res_vehicle);                                                        // [b] Initialise target

            Function.Call(Hash.SET_VEHICLE_OUT_OF_CONTROL,
                target.vehicle,
                true,                                               // a - Driver dies
                false);                                             // b - Vehicle does not explode on impact

            return target;
        }

        public static Target ReduceGrip(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                 // [a] Select random moving vehicle
            Target target = Target.Setup(runcontrol, res_vehicle);                                                        // [b] Initialise target
             
            Function.Call(Hash.SET_VEHICLE_REDUCE_GRIP, target.vehicle, true);                                            // [c] Cause vehicle to lose grip

            return target;
        }

        public static Target ExplodeTyre(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                 // [a] Select random moving vehicle
            Target target = Target.Setup(runcontrol, res_vehicle);                                                        // [b] Initialise target

            // [c] Explode or deflate tyre

            VehicleTyre random_tyre = Damage.tyres.OrderBy(x => runcontrol.random.Next()).Take<VehicleTyre>(1).ToList()[0];                                                // i: Ascertain corresponding tyre 

            Function.Call(Hash.SET_VEHICLE_TYRE_BURST,
                target.vehicle,
                (int)random_tyre,      // Which tyre bursts 
                false,                 // On rim?
                1000f);                // Extent?

            StopVehicle(runcontrol, target);                                                                              // [d] Stop vehicle within stopping distance

            return target;
        }

        public static Target SteerOff(RunControl runcontrol, Environment environment)
        {
            Vehicle res_vehicle = VehicleSelection.Random(runcontrol, environment, true);                                // [a] Select random moving vehicle
            Target target = Target.Setup(runcontrol, res_vehicle);                                                       // [b] Initialise target

            float lockedangle = (float)(0.5 * (1 + runcontrol.random.NextDouble()));                                     // [c] Steering angle magnitude
            float sign = (-0.5 + runcontrol.random.NextDouble() < 0) ? -1f : 1f;                                         // [d] Steering angle sign 

            Function.Call(Hash.SET_VEHICLE_STEER_BIAS, sign * lockedangle);                                              // [e] Apply steering angle
            StopVehicle(runcontrol, target);                                                                             // [f] Stop vehicle within stopping distance

            return target;
        }

        // TYPE 4: Lane change ===============================================================================================================================

        public static void LaneChange(RunControl runcontrol, Environment environment, Target target)
        {

        }

        // TYPE 5: Right-turn oncoming =======================================================================================================================

        public static void RightTurnOncoming(RunControl runcontrol, Environment environment, Target target)
        {

        }

        // TYPE 6: SPEED =====================================================================================================================================

        public static void Speed(RunControl runcontrol, Environment environment, Target target)
        {
            float vehiclespeed = Function.Call<float>(Hash._GET_VEHICLE_SPEED, target.vehicle);
        }

        // Additional Methods

        public static void StopVehicle(RunControl runcontrol, Target target)
        {
            float stoppingdistance = 10f + (float)(20 * runcontrol.random.NextDouble());
            Function.Call(Hash._TASK_BRING_VEHICLE_TO_HALT,
                target.vehicle,
                stoppingdistance);
        }

        // DAMAGE

        /*
    public static class Damage
    {
        // 1: Choice of damage types ===========================================================================
       
        public static IList<DamageType> controldamagetypes = Enum.GetValues(typeof(DamageType)).Cast<DamageType>().ToList();

        public static IList<DamageType> Choice(RunControl runcontrol, int maxtypes)
        {
            /// A: Initialise damage type list

            List<DamageType> possibledamagetypes = new List<DamageType>();
            possibledamagetypes.AddRange(controldamagetypes);

            /// B: Create list of selected damage types

            int n_items = runcontrol.random.Next(0, maxtypes + 1);         // [a] Number of damage types to apply

            IList<DamageType> random_types = possibledamagetypes.OrderBy(x => runcontrol.random.Next()).Take<DamageType>(n_items).ToList();
            return random_types;
        }

        // 2: Cause selected damage types =======================================================================

        public static IDictionary<DamageType, Action<RunControl, Target>> DamageFunctions =
            new Dictionary<DamageType, Action<RunControl, Target>>{
                { DamageType.Tyres, Tyres},                                // 1: Burst or deflated tyres
                { DamageType.Windows, Windows},                            // 2: Smash windows
                { DamageType.Doors, Doors}                                 // 3: Damage doors
            };

        public static int numdamagetypes = Enum.GetNames(typeof(DamageType)).Length;

        public static Target Cause(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {
            /// A: Select vehicle AND initialise dummy

            Vehicle vehicle = VehicleSelection.Create(runcontrol, environment);                    // [a] Create random vehicle 
            Target target = Target.Setup(runcontrol, vehicle);                                     // [b] Initialise target AND set properties

            /// B: Initialise result string AND select damage types to apply

            string[] stringx = new string[numdamagetypes];                                         // [a] Empty result string
            Other.Populate(stringx, "0");

            IList<DamageType> damagetypes = Choice(runcontrol, 2);                                 // [b] Choose damage types

            /// C: Apply damage

            foreach (DamageType damagetype in damagetypes)                                     
            {
                Damage.DamageFunctions[damagetype](runcontrol, target);                        // [a] Apply specific damage type
                int res_index = controldamagetypes.IndexOf(damagetype);                        // [b] Obtain damage type's result string index
                stringx[res_index] = "1";                                                      // [c] Update result string
            }

            /// D: Update test control incident

            testcontrol.incident = String.Join<string>("", stringx);

            return target;
        }

        // 3: Damage methods [Controlled] ================================================================================

        /// A: Burst tyres 

       

        public static void Tyres(RunControl runcontrol, Target target)
        {
            //// i: Possible/remaining window indices

            IList<VehicleTyre> possibletyres = PossibleTyres(target.vehicle);                                                                          
            int numtyres = possibletyres.Count;

            //// ii: Intialise result string

            string[] res_string = new string[numtyres];
            Other.Populate<string>(res_string, "0");

            //// iii: Tyres to burst

            int tyre_nitems = runcontrol.random.Next(1, numtyres + 1);                                         // [a] Number of tyres to damage
            IList<VehicleTyre> random_tyres = 
                possibletyres.OrderBy(x => runcontrol.random.Next()).Take<VehicleTyre>(tyre_nitems).ToList();  // [b] Tyres to be damaged

            //// iv: Burst tyres

            foreach (VehicleTyre random_tyre in random_tyres)
            {
                // [a] Cause tyre to deflate/burst

                Function.Call(Hash.SET_VEHICLE_TYRE_BURST,
                    target.vehicle,        // Vehicle
                    (int)random_tyre,      // Which tyre bursts 
                    false,                 // On rim?
                    1000f);                // Extent

                // [b] Update result string

                int res_index = possibletyres.IndexOf(random_tyre);
                res_string[res_index] = "1";
            }

            //// v: Update target properties

            target.tyresburst = String.Join<string>("", res_string);                                      
        }

        /// B: Smash window         



        public static void Windows(RunControl runcontrol, Target target)
        {
            //// i: Possible windows

            IList<VehicleWindow> possiblewindows = PossibleWindows(target.vehicle);                                                                          
            int numwindows = possiblewindows.Count;

            //// ii: Intialise result string

            string[] res_string = new string[numwindows];
            Other.Populate<string>(res_string, "0");

            //// iii: Windows ro smash

            int window_nitems = runcontrol.random.Next(1, numwindows + 1);                                      // [a] Select how many windows to break
            IList<VehicleWindow> random_windows = possiblewindows.OrderBy(x => runcontrol.random.Next()).
                Take<VehicleWindow>(window_nitems).ToList();                                                    // [b] Select specified windows to break

            //// iv: Smash selected windows

            foreach (VehicleWindow random_window in random_windows)
            {
                Function.Call(Hash.SMASH_VEHICLE_WINDOW, target.vehicle, (int)random_window);                   // [a] Smash window

                int res_index = possiblewindows.IndexOf(random_window);                                         // [b] Update result string
                res_string[res_index] = "1";
            }

            //// v: Update target properties

            target.windowsbroken = String.Join<string>("", res_string);
        }

        /// C: Remove door

        public static void Doors(RunControl runcontrol, Target target)
        {
            //// i: Possible doors

            IList<VehicleDoor> possibledoors = target.vehicle.GetDoors().ToList<VehicleDoor>();                                              
            int numdoors = possibledoors.Count;
            
            //// ii: Intialise result string

            string[] res_string = new string[numdoors];
            Other.Populate<string>(res_string, "0");

            //// iii: Select doors to remove

            int door_nitems = runcontrol.random.Next(1, numdoors + 1);                                          // [a] Select how many doors to damage
            IList<VehicleDoor> random_doors = possibledoors.OrderBy(x => runcontrol.random.Next())
                .Take<VehicleDoor>(door_nitems).ToList();                                                       // [b] Select specified number of doors to break 

            //// iv: Break selected number of doors

            foreach (VehicleDoor random_door in random_doors)
            {
                Function.Call(Hash.SET_VEHICLE_DOOR_BROKEN,                                                      // [a] Break door
                    target.vehicle,
                    (int)random_door,
                    true);

                int res_index = possibledoors.IndexOf(random_door);                                              // [b] Update result string
                res_string[res_index] = "1";
            }

            //// v: Update target properties

            target.doorsdamaged = String.Join<string>("", res_string);
        }

        /// D: Break bumper
        
        public static void Bumpers(Target target)
        {

        }


    }

    public static class DamageMethodsUncontrol
    { 
        // A: EXPLODE CAR =========================================================================================================

        public static void ExplodeCar(RunControl runcontrol, Target target)
        {
            Function.Call(Hash.EXPLODE_VEHICLE,
                target.vehicle,                   // Target vehicle
                true,                             // Is explosion audible?
                false);                           // Is explosion invisible?
        }

        // B: SET DAMAGE ==========================================================================================================

        public static void SetDamage(RunControl runcontrol, Target target)
        {
            // a: Random variation

            float maxoffset = 2f;
            float xoffset = (float)(maxoffset * runcontrol.random.NextDouble());
            float yoffset = (float)(maxoffset * runcontrol.random.NextDouble());
            Vector3 damagelocation = target.vehicle.Position + new Vector3(xoffset, yoffset, 0);

            float damage = (float)(500 * (1 + runcontrol.random.NextDouble()));

            float maxradius = 5f;
            float radius = (float)(maxradius * runcontrol.random.NextDouble());

            // b: Apply damage

            target.vehicle.ApplyDamage(damagelocation, damage, radius);
        }
    }
    */

    }


}
