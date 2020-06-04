using System;
using GTA;

namespace DRS
{
    public static class Operation
    {
        public static void Test(RunControl runcontrol)
        {
            // Function: Run a single experiment
            // Output: 1 x wide jpeg/json, 4 x close jpeg/json

            TestControl testcontrol = TestControl.Setup();                         // [a] Setup simulation
            Environment environment = Environment.Setup(runcontrol, testcontrol);  // [b] Initialise environment AND move player to the drone base
            Collision.Cause(runcontrol, testcontrol);                              // [c] Cause a collision between a target and colliding vehicle

            Response.Capture(runcontrol, testcontrol, environment);                // [d] Capture wide image: image, json

            if (testcontrol.iswidecaptured)
            {
                Response.CaptureDamagedVehicles(runcontrol, testcontrol, environment); // [e] Capture damaged target vehicles: 4 x image, 4 x json
                testcontrol.TestUpdate();                                              // [f] Update time; Delete damaged, target and colliding vehicles; Send DB
            }

            TestControl.DeleteDamagedVehicles(runcontrol);
            RunControl.RenderCreatedCameras(false);                                // [g] Switch to player camera
            Script.Wait(2000);                                                     // [h] Create distinction between successive tests
        }

        public static void Run(RunControl runcontrol)
        {            
            // Function: Manage simulation
            // Output: SQL - runcontrol

            for (double i = 1; i <= runcontrol.iterations; i++)
            {
                Test(runcontrol);                                                  // [a] Run test
                Progress(i, runcontrol.iterations);                                // [b] Record progress
            }
            
            runcontrol.ToDB();                                                     // [c] Write updated runcontrol to SQL database
            runcontrol.ResetPlayerAtMainBase();                                    // [d] Reset player at the main base
        }
        public static void Progress(double icurrent, int imax)
        {
            // Function: Record simulation progress
            // Output: % progression (to screen)

            double progress_int = Math.Round((icurrent / imax) * 100);
            string progress_str = icurrent.ToString() + " [" + progress_int.ToString() + "%]";
            UI.ShowSubtitle(progress_str, 1000);
        }
    }
}
