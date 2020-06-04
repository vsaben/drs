using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public static class Collision
    {
        public static void Cause(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Cause a collision between a target and colliding vehicle; Wait for the collision to end

            SelectTarget(runcontrol, testcontrol);                                                              // [a] Select target and insert player into any seat
            Create(runcontrol, testcontrol);                                                                    // [b] Create collision between target and colliding vehicle
            PostControl(testcontrol);                                                                           // [c] Wait for collision event to end 
        }

        public static void SelectTarget(RunControl runcontrol, TestControl testcontrol)
        {            
            testcontrol.target_vehicle = VehicleSelection.Random(runcontrol, testcontrol);                            // [a] Choose target car            
            Game.Player.Character.SetIntoVehicle(testcontrol.target_vehicle, VehicleSeat.Driver); Script.Wait(1000);  // [b] Position player into vehicle
        }

        public static Dictionary<string, float> COLLISION_PARAMS = new Dictionary<string, float>()
        {
            {"SR",  0.5f },  // Start radius
            {"IMS", 20f },   // Initial minimum speed
            {"IES", 100f }   // Initial excess speed
        };
        public static void SetInitialSpeed(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Set the initial speed of the colliding vehicle

            float initialspeed = (float)(COLLISION_PARAMS["IMS"] + COLLISION_PARAMS["IES"] * runcontrol.random.NextDouble());
            testcontrol.colliding_vehicle.Speed = initialspeed;
        }

        public static void Create(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Create collision between the target and colliding vehicle

            /// A: Determine colliding vehicle heading and starting position

            float angleofimpact = (float)(runcontrol.random.NextDouble() * 2 * Math.PI);                        // [a] Select random angle of impact into target vehicle (in radians)

            Vector3 cv_pos = testcontrol.target_vehicle.Position +                                              // [b] Colliding vehicle starting position
                COLLISION_PARAMS["SR"] * testcontrol.target_vehicle.ForwardVector * (float)Math.Cos(angleofimpact) +
                COLLISION_PARAMS["SR"] * (float)Math.Sin(angleofimpact) * testcontrol.target_vehicle.RightVector;

            float cv_heading = (testcontrol.target_vehicle.Position - cv_pos).ToHeading();                      // [c] Colliding vehicle heading

            /// B: Create collision 

            Vehicle[] nearbyvehicles = World.GetNearbyVehicles(testcontrol.target_vehicle.Position, 100f);      // [d] Freeze nearby vehicles 
            nearbyvehicles.Select(x => x.FreezePosition = true);
            testcontrol.target_vehicle.Speed = 0;                                                               // [e] Stop target vehicle (to enable collision)
            
            testcontrol.colliding_vehicle = VehicleSelection.Create(runcontrol, cv_pos, cv_heading);            // [f] Create colliding vehicle

            while (testcontrol.colliding_vehicle is null)
            {
                testcontrol.colliding_vehicle = VehicleSelection.Create(runcontrol, cv_pos, cv_heading);
            }

            SetInitialSpeed(runcontrol, testcontrol);                                                           // [g] Set colliding vehicle initial speed
            nearbyvehicles.Select(x => x.FreezePosition = false);                                               // [h] Unfreeze nearby vehicles
        }

        public static void PostControl(TestControl testcontrol)
        {
            // Function - Output: Freeze target and collider after the collision "ends"   

            int i = 0;
            while (EndCheck(testcontrol.target_vehicle, i) | EndCheck(testcontrol.colliding_vehicle, i))
            {
                Script.Wait(1000);                             
                i += 1;
            }

            List<Vehicle> damaged_vehicles = World.GetNearbyVehicles(testcontrol.target_vehicle.Position, 1000f)
                .Where(x => Damage.DamageCheck(x)).ToList();
            
            damaged_vehicles.ForEach(x => x.FreezePosition = true); 
            damaged_vehicles.ForEach(x => x.IsPersistent = true);
        }

        public static bool EndCheck(Vehicle vehicle, int i)
        {
            // Function - Output: Check if the collision is still ongoing 

            bool moving = !vehicle.IsStopped;           // [a] Check 1: If the vehicle is still moving
            bool activetime = (i <= 7);                 // [b] Check 2: If less than 7 seconds have passed

            bool res_check = moving && activetime;
            return res_check;
        }
    }   
}
