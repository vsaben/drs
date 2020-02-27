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
        public static void Cause(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {
            testcontrol.target_vehicle = VehicleSelection.Random(runcontrol, environment);     // [a] Choose target car
            Script.Wait(1);
            Game.Player.Character.SetIntoVehicle(testcontrol.target_vehicle, VehicleSeat.Any); // [b] Position player into vehicle
            Script.Wait(1);

            Create(runcontrol, testcontrol);                                                   // [c] Create collision between target and colliding vehicle
            PostControl(testcontrol);                                                          // [d] Wait for collision event to end 
        }

        // 1: Functions ======================================================================================================

        /// A: Create collision between target and colliding vehicle

        public static float START_RADIUS = 0.5f;
        public static float INITIAL_MIN_SPEED = 20f;
        public static float INITIAL_EXC_SPEED = 140f;

        public static void SetInitialSpeed(RunControl runcontrol, TestControl testcontrol)
        {
            float initialspeed = (float)(INITIAL_MIN_SPEED + INITIAL_EXC_SPEED * runcontrol.random.NextDouble());
            testcontrol.colliding_vehicle.Speed = initialspeed;
            testcontrol.initialspeed = initialspeed;
        }

        public static void Create(RunControl runcontrol, TestControl testcontrol)
        {
            Vehicle target_vehicle = testcontrol.target_vehicle;

            //// i: Determine colliding vehicle heading and starting position

            testcontrol.angleofimpact = (float)(runcontrol.random.NextDouble() * 2 * Math.PI);                  // [a] Angle of impact into target vehicle

            Vector3 cv_pos = target_vehicle.Position +                                                          // [b] Colliding vehicle starting position
                START_RADIUS * target_vehicle.ForwardVector * (float)Math.Cos(testcontrol.angleofimpact) +
                START_RADIUS * (float)Math.Sin(testcontrol.angleofimpact) * target_vehicle.RightVector;

            float cv_heading = (target_vehicle.Position - cv_pos).ToHeading();                                  // [c] Colliding vehicle heading

            //// ii: Create colliding vehicle and set its initial speed 

            testcontrol.colliding_vehicle = VehicleSelection.Create(runcontrol, cv_pos, cv_heading);
            SetInitialSpeed(runcontrol, testcontrol);
        }

        /// B: Allow collision consequences to take effect  

        public static void PostControl(TestControl testcontrol)
        {
            //// i: Wait for both cars to stop or for 5 seconds to pass 

            int i = 0;
            while (EndCheck(testcontrol.target_vehicle, i) | EndCheck(testcontrol.colliding_vehicle, i))
            {
                Script.Wait(1000);                             // [a] Wait 1 sec 
                i += 1;
            }

            //// ii: Ensure car speed is 0

            testcontrol.target_vehicle.Speed = 0f;
            testcontrol.colliding_vehicle.Speed = 0f;
        }

        public static bool EndCheck(Vehicle vehicle, int i)
        {
            // Conditions: Continue waiting

            bool moving = !vehicle.IsStopped;
            bool activetime = (i <= 5);

            bool res_check = moving && activetime;
            return res_check;
        }
    }   
}
