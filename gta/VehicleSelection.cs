using System;
using System.Collections.Generic;
using System.Linq;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public static class VehicleSelection
    {
        // 1: Select random or create vehicle ===================================================================================================

        public static Vehicle Random(RunControl runcontrol, Vehicle[] nearbyvehicles)
        {
            // Function - Output: Select random allowable vehicle instance within 5km of the selected drone base 
                                                                                                       // [b] Allowable vehicle types/model
            int nvehicle = nearbyvehicles.Length;
            if (nvehicle > 0)
            {
                Vehicle res_vehicle = nearbyvehicles[runcontrol.random.Next(nvehicle)];                                // [c] Select random vehicle
                res_vehicle.MakeCollisionReady();
                return res_vehicle;
            }
            return null;
        }

        public static void MakeCollisionReady(this Vehicle vehicle)
        {
            vehicle.SetPersistence(true);
            vehicle.Passengers.ToList().ForEach(x => x.Delete());                                              // [d] Set vehicle as persistent
            vehicle.Driver.Delete();
            vehicle.CanBeVisiblyDamaged = true;                                                                // [e] Set vehicle as can be visibly damaged   
            vehicle.IsCollisionProof = false;
        }

        public static Vehicle Collider(RunControl runcontrol, TestControl testcontrol, Vehicle[] nearbyvehicles, Vector3 carposition, float heading = 0f) 
        {
            Vehicle[] validvehicles = nearbyvehicles.Where(x => ALLOWED_VEHICLE_CLASSES.Contains(x.ClassType) 
                                                                & !ERRONEOUS_VEHICLE_MODELS.Contains(x.Model) 
                                                                & !x.Equals(testcontrol.target_vehicle)).ToArray();

            int nvehicle = validvehicles.Length;

            if (nvehicle > 0)
            {
                Vehicle collider = validvehicles[runcontrol.random.Next(nvehicle)];
                collider.MakeCollisionReady();
                collider.Position = carposition;
                collider.Heading = heading;
                collider.FreezePosition = false;
                return collider;
            }

            return null;
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

            res_vehicle.SetPersistence(true);                                                                           // [d] Set car as persistent
            UI.Notify("created vehicle");
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

        public static List<Model> ERRONEOUS_VEHICLE_MODELS = new List<Model>
        {
            new Model(VehicleHash.Buccaneer),
            new Model(VehicleHash.Camper),            
            new Model(VehicleHash.Panto), 
            new Model(VehicleHash.Pounder),
            new Model(VehicleHash.RentalBus),
            new Model(VehicleHash.Taco),
            new Model(VehicleHash.Vigero),
            new Model(VehicleHash.XA21),
            new Model(VehicleHash.Youga), 
            new Model(VehicleHash.Youga2)
        };
    }
}
