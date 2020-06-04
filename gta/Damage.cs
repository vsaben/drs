using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public class Damage
    {
        public int id;                             // Damage ID

        public Dictionary<string, string> body;    // Body: "tyresburst", "doorsdamaged", "windowsbroken", "bumpersbroken" (string representation)
        public Dictionary<string, int> health;     // Health: "driveable", "vehicle", "body"
        public Dictionary<string, bool> extra;     // Extra: "onroof", "onallwheels", "isonfire" 

        // 1: Setup =============================================================================
        public static Damage Setup(int damage_id, Vehicle vehicle)
        {
            Damage damage = new Damage()
            {
                id = damage_id,

                body = new Dictionary<string, string>()
                {
                    { "tyresburst", AreTyresBurst(vehicle) },
                    { "doorsdamaged", AreDoorsDamaged(vehicle) },
                    { "windowsbroken", AreWindowsBroken(vehicle) },
                    { "bumpersbroken", AreBumpersBroken(vehicle) }
                },

                health = new Dictionary<string, int>()
                {
                    { "driveable", vehicle.IsDriveable.BoolToInt() },
                    { "vehiclehealth", vehicle.Health },
                    { "bodyhealth", (int)vehicle.BodyHealth }
                },

                extra = new Dictionary<string, bool>()
                {
                    { "onroof", Function.Call<bool>(Hash.IS_VEHICLE_STUCK_ON_ROOF, vehicle) },
                    { "onallwheels", vehicle.IsOnAllWheels },
                    { "isonfire", vehicle.IsOnFire }
                }
            };
                
            return damage;
        }

        // 2: Check for component damage ========================================================

        // Output: Variable-sized (excl. bumpers) binary string representation 

        public static string AreTyresBurst(Vehicle vehicle)
        {
            IList<VehicleTyre> possibletyres = vehicle.PossibleTyres();
            int numtyres = possibletyres.Count;

            string[] res_string = new string[numtyres];
            Convert.Populate<string>(res_string, "0");

            foreach (VehicleTyre tyre in possibletyres)
            {
                bool is_burst = vehicle.IsTireBurst((int)tyre);

                if (is_burst)
                {
                    int res_index = possibletyres.IndexOf(tyre);
                    res_string[res_index] = "1";
                }
            }

            string combined_res_string = String.Join<string>("", res_string);
            return combined_res_string;
        }

        public static string AreDoorsDamaged(Vehicle vehicle)
        {
            List<VehicleDoor> possibledoors = vehicle.PossibleDoors().ToList();
            int numdoors = possibledoors.Count;

            string[] res_string = new string[numdoors];
            Convert.Populate<string>(res_string, "0");

            foreach (VehicleDoor door in possibledoors)
            {
                bool is_damaged = Function.Call<bool>(Hash.IS_VEHICLE_DOOR_DAMAGED, vehicle, (int)door);

                if (is_damaged)
                {
                    int res_index = possibledoors.IndexOf(door);
                    res_string[res_index] = "1";
                }
            }

            string combined_res_string = String.Join<string>("", res_string);
            return combined_res_string;
        }

        public static string AreWindowsBroken(Vehicle vehicle)
        {
            IList<VehicleWindow> possiblewindows = vehicle.PossibleWindows();
            int numwindows = possiblewindows.Count;

            string[] res_string = new string[numwindows];
            Convert.Populate<string>(res_string, "0");

            foreach (VehicleWindow window in possiblewindows)
            {
                bool is_damaged = !Function.Call<bool>(Hash.IS_VEHICLE_WINDOW_INTACT, vehicle, (int)window);

                if (is_damaged)
                {
                    int res_index = possiblewindows.IndexOf(window);
                    res_string[res_index] = "1";
                }
            }

            string combined_res_string = String.Join<string>("", res_string);
            return combined_res_string;
        }
        public static string AreBumpersBroken(Vehicle vehicle)
        {
            string[] res_string = new string[2];
            Convert.Populate<string>(res_string, "0");

            res_string[0] = vehicle.IsFrontBumperBrokenOff.BoolToIntString();
            res_string[1] = vehicle.IsRearBumperBrokenOff.BoolToIntString();

            string combined_res_string = String.Join<string>("", res_string);
            return combined_res_string;
        }

        // 3: Vehicle damage status check =================================================================
        public static bool DamageCheck(Vehicle vehicle)
        {
            // Function: Check if any body or tyre damage
            // Output: Bool

            bool nativedamagecheck = vehicle.IsDamaged;
            bool istyreburst = TyreBurstCheck(vehicle);           // [a] Unable to check for puncture        
            bool isbodydamaged = vehicle.BodyHealth < 1000f;      // [b] Check: Body damage

            return nativedamagecheck | isbodydamaged | istyreburst;
        }
        public static bool TyreBurstCheck(Vehicle vehicle)
        {
            // Function: Check if any tyre has burst
            // Output: Bool

            IList<VehicleTyre> possibletyres = vehicle.PossibleTyres();

            foreach (VehicleTyre tyre in possibletyres)
            {
                bool is_burst = vehicle.IsTireBurst((int)tyre);
                if (is_burst) return is_burst;
            }
            return false;
        }
        public static bool WindowsBrokenCheck(Vehicle vehicle)
        {
            // Function: Check if any windows are broken
            // Output: Bool

            IList<VehicleWindow> possiblewindows = vehicle.PossibleWindows();

            foreach (VehicleWindow window in possiblewindows)
            {
                bool is_broken = !Function.Call<bool>(Hash.IS_VEHICLE_WINDOW_INTACT, vehicle, (int)window);
                if (is_broken) return is_broken;
            }

            return false;
        }
    }

}
