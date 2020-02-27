using System;
using GTA;


namespace DRS
{
    public static class Operation
    {
        // 1: SIMULATION ====================================================================================== 

        public static void Test(RunControl runcontrol)
        {  
            /// A: Simulation
            
            TestControl testcontrol = TestControl.Setup();                         // [a] Setup simulation
            Environment environment = Environment.Setup(runcontrol);               // [b] Initialise environment AND move player to the drone base

            /// B: Collision

            Collision.Cause(runcontrol, testcontrol, environment);                 // [c] Cause a collision between a target and colliding vehicle
            
            /// C: Response

            Response.PositionWideCam(runcontrol, testcontrol);                     // [d] Position camera at random location and directional offset to target 

            WriteToTiff.PrepareGameBuffer(true);
            testcontrol.Update(runcontrol, environment);

            UI.Notify("Vehicles (Damaged): " + testcontrol.numvehicles.ToString() + " (" + testcontrol.numdamaged.ToString() + ")");

            Response.TakePicture(testcontrol);                                     // [d] Take a picture of wide collision area
            TestControl.CaptureVehicles(runcontrol, testcontrol);                  // [e] Capture vehicle information 
                                                                                   //      - Take a variable 4-picture sequence of damaged cars  
                                                                                   //      - Delete damaged vehicles
            /// B: Update control and save to database

            WriteToTiff.PrepareGameBuffer(false);
            Script.Wait(2000);                                                     // [c] Create distinction between successive tests
        }

        // 4: Overall: Run ======================================================================================

        public static void Run(RunControl runcontrol)
        {
            /// A: Run tests

            for (double i = 1; i <= runcontrol.iterations; i++)
            {
                Operation.Test(runcontrol);                       // [a] Run test
                Progress(i, runcontrol.iterations);               // [b] Record progress
            }

            /// B: Update RUNCONTROL database
            
            // runcontrol.Update();                                  // [a] Update runcontrol properties
            // runcontrol.ToDB();                                    // [b] Write to RUNCONTROL database
            runcontrol.ResetPlayerAtMainBase();
        }

        // 3: Functions ===========================================================================================
        public static void Progress(double icurrent, int imax)
        {
            double progress_int = Math.Round((icurrent / imax) * 100);
            string progress_str = icurrent.ToString() + " [" + progress_int.ToString() + "%]";
            UI.ShowSubtitle(progress_str, 1000);
        }


    }
}
