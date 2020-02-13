using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;

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

        public static Vector3 playeroffset = new Vector3(5f, 5f, 0f);                           // [a] Player initialisation positioning offset 

        public static RunControl Setup(int iter)
        {
            Game.Player.Character.Position = Environment.mainbaseposition + playeroffset;      
            Script.Wait(2000);                                                                  // [a] Allow initial base to load 

            /// A: Initialise camera

            Camera cam = World.CreateCamera(
                Environment.mainbaseposition,                                                   // [a] Location
                new Vector3(0f, 0f, 0f),                                                        // [b] Offset
                50f);                                                                           // [c] Field of view                     

            Function.Call(Hash.RENDER_SCRIPT_CAMS, 0, 1, 0, 0, 0);

            /// B: RunControl class setup            

            RunControl runcontrol = new RunControl()
            {                
                iterations = iter,
                startdatetime = System.DateTime.Now,
                random = new Random(),
                camera = cam,
            };

            runcontrol.id = DB.LastID("RunControl") + 1;

            int nexttestid = DB.LastID("TestControl") + 1;
            runcontrol.testidrange = nexttestid.ToString();

            /// C: Turn off game features

            GameplayOptions(true);

            return runcontrol;
        }

        public static void Update(RunControl runcontrol)
        {
            /// A: ID Ranges

            runcontrol.testidrange = runcontrol.testidrange + "-" + DB.LastID("TestControl").ToString();

            // B: Time

            runcontrol.enddatetime = System.DateTime.Now;
            runcontrol.duration = (runcontrol.enddatetime - runcontrol.startdatetime).Duration();
        }

        // 2: Methods ==============================================================================

        public static void GameplayOptions(bool turnoff)
        {
            Game.Player.Character.IsVisible = !turnoff;           // Make player invisible
            Game.Player.Character.IsInvincible = turnoff;         // Player cannot be killed
            Game.Player.IgnoredByPolice = turnoff;                // Player is ignored by the police
            if (turnoff)
            {
                Game.MaxWantedLevel = 0;                          // Maximum wanted level is 0
            }
        }

        public static void ResetPlayerAtBase()
        {
            Function.Call(Hash.RENDER_SCRIPT_CAMS, 0, 1, 0, 0, 0);         // [a] Camera = Player camera

            Game.Player.Character.Position = Environment.mainbaseposition; // [b] Return player to the main base
            Game.Player.Character.IsVisible = true;                        // [c] Make player visible                                                                        
        }
    }
}
