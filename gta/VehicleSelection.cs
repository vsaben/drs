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
            Vehicle[] nearby_vehicles = World.GetNearbyVehicles(environment.baseposition, 5000f).              /// [a] Nearby vehicles
                Where(x => AllowableVehicleClassCheck(x.Model)).ToArray<Vehicle>();                            /// [b] Allowable vehicle types

            Vehicle res_vehicle = nearby_vehicles[runcontrol.random.Next(nearby_vehicles.Length)];             /// [c] Select random vehicle

            res_vehicle.IsPersistent = true;                                                                   /// [d] Set vehicle as persistent
            res_vehicle.CanBeVisiblyDamaged = true;                                                            /// [e] Set vehicle as can be visibly damaged

            return res_vehicle;
        }

        /// B: Create vehicle of an allowable vehicle class 

        public static Vehicle Create(RunControl runcontrol, Vector3 carposition, float heading = 0f)
        {            
            Model randomcarmodel = AllPossibleVehicleModels[runcontrol.random.Next(AllPossibleVehicleModels.Count)];   // [a] Random car model
            Vehicle res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);                           // [b] Create car 

            while (res_vehicle is null)                                                                        // [c] Some car models are not created, create another 
            {                                                                                                  
                res_vehicle = World.CreateVehicle(randomcarmodel, carposition, heading);               
            }

            res_vehicle.IsPersistent = true;                                                                   // [d] Set car as persistent
            Function.Call(Hash.REQUEST_VEHICLE_HIGH_DETAIL_MODEL, res_vehicle);                                // [e] High detail model

            return res_vehicle;
        }

        // 2: Parameters =========================================================================================

        public static IList<VehicleClass> AllowableVehicleClasses = new List<VehicleClass>
        {
            VehicleClass.Commercial,
            VehicleClass.Compacts,
            VehicleClass.Coupes,
            VehicleClass.Emergency,
            VehicleClass.Industrial,
            VehicleClass.Muscle,
            VehicleClass.Sedans,
            VehicleClass.Service,
            VehicleClass.Sports,
            VehicleClass.SportsClassics,
            VehicleClass.Super,
            VehicleClass.SUVs, 
            VehicleClass.Utility,
            VehicleClass.Vans
        };

        public static bool AllowableVehicleClassCheck(Model model)
        {
            VehicleClass vehicleclass = (VehicleClass)Function.Call<int>(Hash.GET_VEHICLE_CLASS_FROM_NAME, model.Hash);
            bool res_class = AllowableVehicleClasses.Contains(vehicleclass);
            return res_class;
        }

        public static IList<VehicleHash> AllPossibleVehicleHashes = Enum.GetValues(typeof(VehicleHash)).OfType<VehicleHash>()
            .Where(x => AllowableVehicleClassCheck(new Model(x))).ToList<VehicleHash>();

        public static IList<Model> AllPossibleVehicleModels = AllPossibleVehicleHashes.Select<VehicleHash, Model>(x => new Model(x)).ToList<Model>();     
    }
}
