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
        // 1: Functions ==========================================================================================

        /// A: Select random allowable vehicle within 5km of the selected drone base 

        public static Vehicle Random(RunControl runcontrol, Environment environment)
        {                                                        
            //// i: Obtain nearby vehicles

            Vehicle[] nearby_vehicles = World.GetNearbyVehicles(environment.baseposition, 5000f).              /// [a] Nearby vehicles
                Where(x => AllowableVehicleClass(x.Model)).ToArray<Vehicle>();                                 /// [b] Allowable vehicle types

            //// ii: Select (IF nearby vehicle exists) or create vehicle

            Vehicle res_vehicle = nearby_vehicles[runcontrol.random.Next(nearby_vehicles.Length)];             /// [a] Select vehicle

            //// iii: Set vehicle as persistent

            res_vehicle.IsPersistent = true;
            res_vehicle.CanBeVisiblyDamaged = true;

            return res_vehicle;
        }

        /// B: Create "colliding" vehicle of an allowable vehicle class 

        public static Vehicle Create(RunControl runcontrol, Vector3 carposition, float heading = 0f)
        {            
            Model randomcarmodel = AllPossibleCarModels[runcontrol.random.Next(AllPossibleCarModels.Count)];   // [a] Random car model
            Vehicle res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);                   // [b] Create car 

            while (!res_vehicle.Exists())
            {
                res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);               
            }

            res_vehicle.IsPersistent = true;                                                                   // [c] Set car as persistent
            Function.Call(Hash.REQUEST_VEHICLE_HIGH_DETAIL_MODEL, res_vehicle);                                // [d] High detail model

            return res_vehicle;
        }

        // 2: Parameters =====================================================================================

        public static IList<VehicleClass> AllowableVehicleClasses = new List<VehicleClass>
        {
            VehicleClass.Commercial,
            VehicleClass.Compacts,
            VehicleClass.Coupes,
            VehicleClass.Emergency,
            VehicleClass.Industrial,
            VehicleClass.Muscle,
            VehicleClass.Sedans,
            VehicleClass.Sports,
            VehicleClass.SportsClassics,
            VehicleClass.Super,
            VehicleClass.SUVs, 
            VehilceClass.Utility,
            VehicleClass.Vans
        };

        public static bool AllowableVehicleClass(Model model)
        {
            VehicleClass vehicleclass = (VehicleClass)Function.Call<int>(Hash.GET_VEHICLE_CLASS_FROM_NAME, model.Hash);
            bool res_class = AllowableVehicleClasses.Contains(vehicleclass);
            return res_class;
        }

        public static IList<VehicleHash> AllPossibleCarHashes = Enum.GetValues(typeof(VehicleHash)).OfType<VehicleHash>()
            .Where(x => AllowableVehicleClass(new Model(x))).ToList<VehicleHash>();

        public static IList<Model> AllPossibleCarModels = AllPossibleCarHashes.Select<VehicleHash, Model>(x => new Model(x)).ToList<Model>();
        
    }
}
