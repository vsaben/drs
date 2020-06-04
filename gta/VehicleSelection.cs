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
    public static class VehicleSelection
    {
        // 1: Select random or create vehicle ===================================================================================================

        public static Vehicle Random(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Select random allowable vehicle instance within 5km of the selected drone base 

            Vehicle[] nearby_vehicles = World.GetNearbyVehicles(testcontrol.baseposition, 5000f).                      // [a] Nearby vehicles
                Where(x => ALLOWED_VEHICLE_CLASSES.Contains(x.ClassType)).ToArray();                                   // [b] Allowable vehicle types

            Vehicle res_vehicle = nearby_vehicles[runcontrol.random.Next(nearby_vehicles.Length)];                     // [c] Select random vehicle

            while (res_vehicle is null)
            {
                res_vehicle = nearby_vehicles[runcontrol.random.Next(nearby_vehicles.Length)];
            }

            res_vehicle.IsPersistent = true;                                                                           // [d] Set vehicle as persistent
            res_vehicle.CanBeVisiblyDamaged = true;                                                                    // [e] Set vehicle as can be visibly damaged            

            return res_vehicle;
        }
        public static Vehicle Create(RunControl runcontrol, Vector3 carposition, float heading = 0f)
        {
            // Function - Output: Create vehicle of an allowable vehicle class 

            Model randomcarmodel = ALLOWED_VEHICLE_MODELS[runcontrol.random.Next(ALLOWED_VEHICLE_MODELS.Count)];       // [a] Random car model
            Vehicle res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);                           // [b] Create car 

            while (res_vehicle is null)                                                                                // [c] Some car models are not created, create another 
            {                                                                                                  
                res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);               
            }

            res_vehicle.IsPersistent = true;                                                                           // [d] Set car as persistent
            return res_vehicle;
        }

        // 2: Allowable vehicle classes =========================================================================================

        public static List<VehicleClass> ALLOWED_VEHICLE_CLASSES = new List<VehicleClass>
        {
            VehicleClass.Compacts,
            VehicleClass.Coupes,
            VehicleClass.Muscle,
            VehicleClass.Sedans,
            VehicleClass.Sports,
            VehicleClass.SportsClassics,
            VehicleClass.Super,
            VehicleClass.SUVs, 
            VehicleClass.Vans
        };

        public static List<Model> ALLOWED_VEHICLE_MODELS = Enum.GetValues(typeof(VehicleHash)).OfType<VehicleHash>()
            .Where(x => x.AllowableVehicleHash()).Select(x => new Model(x)).ToList();
        public static bool AllowableVehicleHash(this VehicleHash hash)
        {
            VehicleClass vehicleclass = (VehicleClass)Function.Call<int>(Hash.GET_VEHICLE_CLASS_FROM_NAME, new Model(hash));
            return ALLOWED_VEHICLE_CLASSES.Contains(vehicleclass);                        
        }          
    }
}
