using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Data.SqlClient;

using GTA;
using GTA.Math;
using GTA.Native;

namespace DRS
{
    public class RunControl
    {
        // 0: Properties ==========================================================================

        /// A: Run Control Information

        public int id;                            // RunControl ID
        public int iterations;                    // Number of tests
        public string testidrange;                // Test ID range   

        /// B: Time

        public DateTime startdatetime;            // Start of experiment
        public DateTime enddatetime;              // End of experiment
        public TimeSpan duration;                 // Length of time

        /// C: Coding

        public Random random;                     // Randomness generator
        public Camera camera;                     // Camera     

        // 1: Setup ===================================================================================

        public static Vector3 PLAYEROFFSET = new Vector3(5f, 5f, 0f);                           // Default player camera offset 

        public static RunControl Setup(int iter)
        {
            Game.Player.Character.Position = Environment.MAINBASEPOSITION + PLAYEROFFSET;       // [a] Set player at the main base
            Script.Wait(2000);                                                                  // [b] Allow initial base to load 

            /// A: Initialise camera

            Camera cam = World.CreateCamera(                                                     
                Environment.MAINBASEPOSITION,                                                   // Location
                new Vector3(0f, 0f, 0f),                                                        // Offset
                55.0f);                                                                         // Field of view (vertical)                    

            RenderCreatedCameras(false);                                                         

            /// B: RunControl class setup            

            RunControl runcontrol = new RunControl()
            {
                id = DB.LastID("RunControl") + 1,
                iterations = iter,
                testidrange = (DB.LastID("TestControl") + 1).ToString(),
                startdatetime = System.DateTime.Now,
                random = new Random(),
                camera = cam,
            };

            /// C: Turn off game features

            TurnOffGameplayOptions(true);

            return runcontrol;
        }

        public void Update()
        {
            testidrange = testidrange + "-" + DB.LastID("TestControl").ToString();   

            enddatetime = System.DateTime.Now;
            duration = (enddatetime - startdatetime).Duration();
        }

        // 2: Methods ==============================================================================

        public static void TurnOffGameplayOptions(bool turnoff)
        {
            // Function - Output: Turn on/off environment impediments 

            Game.Player.Character.IsVisible = !turnoff;                    // [a] Make player invisible
            Game.Player.Character.IsInvincible = turnoff;                  // [b] Player cannot be killed
            Game.Player.IgnoredByPolice = turnoff;                         // [c] Player is ignored by the police
            if (turnoff) Game.MaxWantedLevel = 0;                          // [d] Maximum wanted level is 0
        }

        public void ResetPlayerAtMainBase()
        {
            RenderCreatedCameras(false);                                   // [a] Camera = Player camera
            Game.Player.Character.Position = Environment.MAINBASEPOSITION; // [b] Return player to the main base
            Game.Player.Character.IsVisible = true;                        // [c] Make player visible                                                                        
        }

        public static void RenderCreatedCameras(bool on)
        {
            if (GameplayCamera.IsRendering != on) return;
            Function.Call(Hash.RENDER_SCRIPT_CAMS, Convert.BoolToInt(on), 1, 0, 0, 0);            
        }


        // 3: Database =============================================================================

        public static string[] db_run_control_parameters =
        {
            "RunControlID",
            "TestIDRange",
            "Iterations",
            "StartDateTime",
            "EndDateTime",
            "Duration"
        };

        public static string sql_run_control = DB.SQLCommand("RunControl", db_run_control_parameters);

        public void ToDB()
        {
            Update();
            SqlConnection cnn = DB.InitialiseCNN();

            using (SqlCommand cmd = new SqlCommand(sql_run_control, cnn))
            {
                cmd.Parameters.AddWithValue("@RunControlID", id);

                cmd.Parameters.AddWithValue("@TestIDRange", testidrange);
                cmd.Parameters.AddWithValue("@Iterations", iterations);
                cmd.Parameters.AddWithValue("@StartDateTime", startdatetime);
                cmd.Parameters.AddWithValue("@EndDateTime", enddatetime);
                cmd.Parameters.AddWithValue("@Duration", duration);

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }

    }
}
