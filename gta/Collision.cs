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

            Vehicle[] nearbyvehicles = World.GetNearbyVehicles(testcontrol.baseposition, 5000f).                      // [a] Nearby vehicles
                Where(x => VehicleSelection.ALLOWED_VEHICLE_CLASSES.Contains(x.ClassType) & 
                           !VehicleSelection.ERRONEOUS_VEHICLE_MODELS.Contains(x.Model)).ToArray();
            nearbyvehicles.ToList().Select(x => x.FreezePosition = true);

            SelectTarget(runcontrol, testcontrol, nearbyvehicles);                                              // [a] Select target and insert player into any seat            
            if (!testcontrol.iscollisioninstances)
            {
                UI.Notify("Target instance is not initialised");
                return;
            };

            SelectCollider(runcontrol, testcontrol, nearbyvehicles);                                            // [b] Create collision between target and colliding vehicle
            if (!testcontrol.iscollisioninstances) {
                UI.Notify("Collision instance is not initialised");
                testcontrol.target_vehicle.Delete();
                return; };

            testcontrol.colliding_vehicle.FreezePosition = false;
            testcontrol.target_vehicle.FreezePosition = false;
            nearbyvehicles.Select(x => x.FreezePosition = false);

            PostControl(testcontrol);                                                                           // [c] Wait for collision event to end 
        }
        public static void SelectTarget(RunControl runcontrol, TestControl testcontrol, Vehicle[] nearbyvehicles)
        {                      
            for (int i = 0; i < 3; i++)
            {
                testcontrol.target_vehicle = VehicleSelection.Random(runcontrol, nearbyvehicles);               // [a] Choose target car

                if ((testcontrol.target_vehicle is null) | !testcontrol.target_vehicle.Exists()) 
                {
                    continue;
                };

                Game.Player.Character.SetIntoVehicle(testcontrol.target_vehicle, VehicleSeat.Driver);           // [b] Position player into vehicle                    
                Script.Wait(2000);

                if (testcontrol.target_vehicle.IsVisible)
                {
                    testcontrol.target_vehicle.Speed = 0;                                                       // [e] Stop target vehicle (to enable collision)
                    testcontrol.iscollisioninstances = true;                    
                    return;
                }
            }          
        }

        public static Dictionary<string, float> COLLISION_PARAMS = new Dictionary<string, float>()
        {
            {"SR",  4f },    // Start radius
            {"IMS", 20f },   // Initial minimum speed
            {"IES", 60f }    // Initial excess speed
        };
        public static void SetInitialSpeed(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Set the initial speed of the colliding vehicle

            float initialspeed = (float)(COLLISION_PARAMS["IMS"] + COLLISION_PARAMS["IES"] * runcontrol.random.NextDouble());
            testcontrol.colliding_vehicle.Speed = initialspeed;
        }

        public static void SelectCollider(RunControl runcontrol, TestControl testcontrol, Vehicle[] nearbyvehicles)
        {
            // Function - Output: Create collision between the target and colliding vehicle

            /// A: Determine colliding vehicle heading and starting position

            float angleofimpact = (float)(runcontrol.random.NextDouble() * 2 * Math.PI);                        // [a] Select random angle of impact into target vehicle (in radians)

            Vector3 cv_pos = testcontrol.target_vehicle.Position +                                              // [b] Colliding vehicle starting position
                COLLISION_PARAMS["SR"] * testcontrol.target_vehicle.ForwardVector * (float)Math.Cos(angleofimpact) +
                COLLISION_PARAMS["SR"] * (float)Math.Sin(angleofimpact) * testcontrol.target_vehicle.RightVector;

            float cv_heading = (testcontrol.target_vehicle.Position - cv_pos).ToHeading();                      // [c] Colliding vehicle heading

            /// B: Create collision 

            Vehicle[] validnearbyvehicles = nearbyvehicles
                .Where(x => !x.Equals(testcontrol.target_vehicle)).ToArray();

            testcontrol.iscollisioninstances = false;

            for (int i = 0; i < 3; i++)
            {
                testcontrol.colliding_vehicle = VehicleSelection.Random(runcontrol, validnearbyvehicles);

                if (!(testcontrol.colliding_vehicle is null) & testcontrol.colliding_vehicle.Exists())
                {
                    testcontrol.colliding_vehicle.Position = cv_pos;
                    testcontrol.colliding_vehicle.Heading = cv_heading;
                    SetInitialSpeed(runcontrol, testcontrol);
                    testcontrol.iscollisioninstances = true;
                    return;
                }
            }
                                                                   // [g] Set colliding vehicle initial speed
                                                                   // [h] Unfreeze nearby vehicles
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
            damaged_vehicles.ForEach(x => x.SetPersistence(true));
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
