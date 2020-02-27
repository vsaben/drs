using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public class Target
    {
        public Vehicle vehicle;               // Target vehicle
        public bool istarget;                 // Whether is target
        public bool iscollider;               // Whether is collider

        /// A: COLOURS

        public int numcolours;                // Number of colours
        public string primarycolour;          // Primary colour of vehicle
        public string secondarycolour;        // Secondary colour of vehicle (if available)

        /// B: NAMES

        public string vehicleclass;           // Class of vehicle 
        public string friendlyname;           // Vehicle model - friendly name

        /// C: PROPERTIES

        public IList<VehicleTyre> possibletyres;     // List of vehicle tyre objects
        public IList<VehicleDoor> possibledoors;     // List of vehicle door objects
        public IList<VehicleWindow> possiblewindows; // List of vehicle window objects

        public int capacity;                  // Number of seats
        public int numwindows;                // Number of windows
        public int numdoors;                  // Number of doors
        public int numtyres;                  // Number of tyres
        public bool hasroof;                  // Does the vehicle have a roof?  
        public string numberplate;            // Vehicle's number plate 

        /// D: LOCALISATION

        public Vector3 location;              // Vehicle location in world co-ordinates
        public Vector3 rotation;              // Vehicle rotation in world co-ordinates
        public Vector3 fwdvec;                // Forward vector
        public Vector3 dimensions;            // Model dimensions in model co-ordinates

        /// E: CHECK: DAMAGE
        
        public Damage damage;                 // Damage subclass (if damaged only)

        // 1: Setup ====================================================================================

        public static Target Setup(Vehicle invehicle, bool istarget = false, bool iscollider = false, int damage_id = 0)
        {
            Target target = new Target()
            {
                vehicle = invehicle,
                istarget = istarget,
                iscollider = iscollider,

                /// A: Colours

                numcolours = invehicle.ColorCombinationCount,
                primarycolour = invehicle.PrimaryColor.ToString(),
                secondarycolour = invehicle.SecondaryColor.ToString(),

                /// B: Names

                vehicleclass = invehicle.ClassType.ToString(),
                friendlyname = invehicle.FriendlyName,

                /// C: Properties

                possibletyres = PossibleTyres(invehicle),
                possibledoors = PossibleDoors(invehicle),
                possiblewindows = PossibleWindows(invehicle),

                capacity = invehicle.PassengerSeats + 1,
                hasroof = invehicle.HasRoof,
                numberplate = invehicle.NumberPlate,

                /// D: Localisation

                location = invehicle.Position,
                rotation = invehicle.Rotation,
                dimensions = invehicle.Model.GetDimensions(),
            };

            target.numtyres = target.possibletyres.Count;
            target.numdoors = target.possibledoors.Count;
            target.numwindows = target.possiblewindows.Count;

            if (damage_id > 0)
            {
                target.damage = new Damage() 
                {
                    id = damage_id,

                    tyresburst = Damage.AreTyresBurst(target),
                    doorsdamaged = Damage.AreDoorsDamaged(target),
                    windowsbroken = Damage.AreWindowsBroken(target),
                    bumpersbroken = Damage.AreBumpersBroken(target),

                    driveable = target.vehicle.IsDriveable,
                    vehiclehealth = target.vehicle.Health,
                    bodyhealth = (int)target.vehicle.BodyHealth,

                    onroof = Function.Call<bool>(Hash.IS_VEHICLE_STUCK_ON_ROOF, target.vehicle),
                    onallwheels = target.vehicle.IsOnAllWheels,

                    isonfire = target.vehicle.IsOnFire
                };                
            }
            
            return target;
        }

        // 2: Functions ================================================================================

        /// A: Vehicle properties

        //// DOORS 

        public static IList<VehicleDoor> PossibleDoors(Vehicle vehicle)
        {
            IList<VehicleDoor> possibledoors = vehicle.GetDoors().OrderBy(x => (int)x).ToList<VehicleDoor>();
            return possibledoors;
        }

        //// TYRES

        public static Array tyres_array = Enum.GetValues(typeof(VehicleTyre));
        public static List<VehicleTyre> tyres = tyres_array.Cast<VehicleTyre>().OrderBy(x => (int)x).ToList();
        public static IList<VehicleTyre> PossibleTyres(Vehicle vehicle)
        {
            IList<VehicleTyre> res_tyres = new List<VehicleTyre>();                                    // [a] Initialised list

            foreach (VehicleTyre tyre in tyres)
            {
                Function.Call(Hash.SET_VEHICLE_TYRE_BURST,                                             // [a] Burst tyre 
                    vehicle,
                    (int)tyre,
                    false,
                    1000f);

                bool isburst = Function.Call<bool>(Hash.IS_VEHICLE_TYRE_BURST,                         // [b] Check if tyre is burst
                    vehicle,
                    (int)tyre);

                if (isburst)
                {
                    Function.Call(Hash.SET_VEHICLE_TYRE_FIXED,                                         // [c] Fix tyre        
                        vehicle,
                        (int)tyre);

                    res_tyres.Add(tyre);                                                               // [d] Add tyre to possible tyre list
                }
            }

            return res_tyres;
        }

        //// WINDOWS

        public static Array windows_array = Enum.GetValues(typeof(VehicleWindow));
        public static List<VehicleWindow> windows = windows_array.Cast<VehicleWindow>().OrderBy(x => (int)x).ToList();
        public static IList<VehicleWindow> PossibleWindows(Vehicle vehicle)
        {
            //// i: Initialised list

            IList<VehicleWindow> res_windows = new List<VehicleWindow>();

            //// ii: Possible windows to damage

            foreach (VehicleWindow window in windows)
            {
                bool intact = Function.Call<bool>(Hash.IS_VEHICLE_WINDOW_INTACT, vehicle, (int)window);
                if (intact) res_windows.Add(window);
            }

            return res_windows;
        }

        // 3: Damage subclass ===========================================================================================
        public class Damage
        {
            public int id;                 // Damage ID

            public string tyresburst;             // String representation of which tyres have burst (if any) 
            public string doorsdamaged;           // String representation of which doors have been damaged (if any)
            public string windowsbroken;          // String representation of which windows are broken (if any)
            public string bumpersbroken;          // String representation of which bumpers are broken (if any)

            public bool driveable;                // Is the car driveable?
            public int vehiclehealth;             // Vehicle's "health" value
            public int bodyhealth;                // Body "health" value

            public bool onroof;                   // Is the vehicle on its roof?
            public bool onallwheels;              // Is the vehicle on all 4 of its wheels?

            public bool isonfire;                 // Is the vehicle on fire?

            /// B: Functions

            //// TYRES

            public static string AreTyresBurst(Target target)
            {
                string[] res_string = new string[target.numtyres];
                Other.Populate<string>(res_string, "0");

                foreach (VehicleTyre tyre in target.possibletyres)
                { 
                    bool is_burst = target.vehicle.IsTireBurst((int)tyre);

                    if (is_burst)
                    {
                        int res_index = target.possibletyres.IndexOf(tyre);
                        res_string[res_index] = "1";
                    }
                }

                string combined_res_string = String.Join<string>("", res_string);
                return combined_res_string;
            }

            //// DOORS
            public static string AreDoorsDamaged(Target target)
            {
                string[] res_string = new string[target.numdoors];
                Other.Populate<string>(res_string, "0");

                foreach (VehicleDoor door in target.possibledoors)
                {

                    bool is_damaged = Function.Call<bool>(Hash.IS_VEHICLE_DOOR_DAMAGED, target.vehicle, (int)door);

                    if (is_damaged)
                    {
                        int res_index = target.possibledoors.IndexOf(door);
                        res_string[res_index] = "1";
                    }
                }

                string combined_res_string = String.Join<string>("", res_string);
                return combined_res_string;
            }

            //// WINDOWS
            public static string AreWindowsBroken(Target target)
            {
                string[] res_string = new string[target.numwindows];
                Other.Populate<string>(res_string, "0");

                foreach (VehicleWindow window in target.possiblewindows)
                {

                    bool is_damaged = !Function.Call<bool>(Hash.IS_VEHICLE_WINDOW_INTACT, target.vehicle, (int)window);

                    if (is_damaged)
                    {
                        int res_index = target.possiblewindows.IndexOf(window);
                        res_string[res_index] = "1";
                    }
                }

                string combined_res_string = String.Join<string>("", res_string);
                return combined_res_string;
            }

            //// BUMPERS

            public static string AreBumpersBroken(Target target)
            {
                string[] res_string = new string[2];
                Other.Populate<string>(res_string, "0");

                res_string[0] = target.vehicle.IsFrontBumperBrokenOff ? "1" : "0";
                res_string[1] = target.vehicle.IsRearBumperBrokenOff ? "1" : "0";

                string combined_res_string = String.Join<string>("", res_string);
                return combined_res_string;
            }
        }
    }

    /*
        // 4: DB (SQL) ===============================================================================


        /// B: Target 

        public static string[] db_target_parameters =
        {
            "TargetID",

            //// i: Features

            "Numberplate",
            "NumColours",
            "PrimaryColour",
            "SecondaryColour",
            "DirtLevel",
            "VehicleClass",
            "FriendlyName",
            "Capacity",
            "NumWindows",
            "NumDoors",
            "NumTyres",
            "HasRoof",
            
            //// ii: Damage
            
            "TyresBurst",
            "DoorsDamaged",
            "WindowsBroken",
            "BumpersBroken",
            "Driveable",
            "VehicleHealth",
            "BodyHealth",
            "OnRoof",
            "OnAllWheels",
            "IsOnFire"

        };

        public static void ToDBTarget(RunControl runcontrol, TestControl testcontrol, Target target, SqlConnection cnn)
        {
            string sql_target = DB.SQLCommand("Target", db_target_parameters);

            using (SqlCommand cmd = new SqlCommand(sql_target, cnn))
            {
                cmd.Parameters.AddWithValue("@TargetID", testcontrol.id);

                //// i: Features

                cmd.Parameters.AddWithValue("@Numberplate", target.numberplate);

                cmd.Parameters.AddWithValue("@NumColours", target.numcolours);
                cmd.Parameters.AddWithValue("@PrimaryColour", target.primarycolour);
                cmd.Parameters.AddWithValue("@SecondaryColour", target.secondarycolour);
                cmd.Parameters.AddWithValue("@DirtLevel", target.dirtlevel);

                cmd.Parameters.AddWithValue("@VehicleClass", target.vehicleclass);
                cmd.Parameters.AddWithValue("@FriendlyName", target.friendlyname);

                cmd.Parameters.AddWithValue("@Capacity", target.capacity);
                cmd.Parameters.AddWithValue("@NumWindows", target.numwindows);
                cmd.Parameters.AddWithValue("@NumDoors", target.numdoors);
                cmd.Parameters.AddWithValue("@NumTyres", target.numtyres);
                cmd.Parameters.AddWithValue("@HasRoof", target.hasroof);

                //// ii: Damage

                cmd.Parameters.AddWithValue("@TyresBurst", target.tyresburst);
                cmd.Parameters.AddWithValue("@DoorsDamaged", target.doorsdamaged);
                cmd.Parameters.AddWithValue("@WindowsBroken", target.windowsbroken);
                cmd.Parameters.AddWithValue("@BumpersBroken", target.bumpersbroken);

                cmd.Parameters.AddWithValue("@Driveable", target.driveable);
                cmd.Parameters.AddWithValue("@VehicleHealth", target.vehiclehealth);
                cmd.Parameters.AddWithValue("@BodyHealth", target.bodyhealth);

                cmd.Parameters.AddWithValue("@OnRoof", target.onroof);
                cmd.Parameters.AddWithValue("@OnAllWheels", target.onallwheels);
                cmd.Parameters.AddWithValue("@IsOnFire", target.isonfire);


                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }

      */  
    
}