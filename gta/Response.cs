using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Collections.Generic;
using System.Linq;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public static class Response
    {
        // 1: Functions =================================================================================================

        /// A: Attach camera at random distance and angle to target vehicle

        public static void PositionWideCam(RunControl runcontrol, TestControl testcontrol)
        {
            for(int i = 0; i < 5; i++)
            {
                Vector3 locationoffset = LocationOffset(runcontrol);                                                           

                //// i: Location offset

                runcontrol.camera.AttachTo(testcontrol.target_vehicle, locationoffset);
                RunControl.RenderCreatedCameras(true);

                runcontrol.camera.PointAt(testcontrol.target_vehicle);
                Script.Wait(1);

                //// ii: Angle offset

                Vector3 directionoffset = DirectionOffset(runcontrol);
                runcontrol.camera.PointAt(testcontrol.target_vehicle, directionoffset);

                //// iii: Check if target vehicle is in line-of-sight

                if (TestControl.LOS(testcontrol.target_vehicle)) { break; }

                UI.Notify(i.ToString()); // ADJUST
            }
            Script.Wait(2000);
        }

        //// ii: Location Offset 

        public static float CAM_MIN_DIST = 10f;                                                    // [a] In all positive upvector space
        public static float CAM_EXC_DIST = 40f;

        public static float AZI_MIN_PI_FACTOR = 1/6f;
        public static float AZI_EXC_PI_FACTOR = 4/6f;

        public static Vector3 LocationOffset(RunControl runcontrol)
        {
            float H = CAM_MIN_DIST + CAM_EXC_DIST * (float)runcontrol.random.NextDouble();
            float theta = (float)(2 * Math.PI * (float)runcontrol.random.NextDouble());
            float azimuth = (float)(AZI_MIN_PI_FACTOR + AZI_EXC_PI_FACTOR * (float)runcontrol.random.NextDouble());

            Vector3 location_offset = H * new Vector3((float)Math.Cos(theta), (float)Math.Sin(theta), (float)Math.Tan(azimuth)); 
            return location_offset;
        }

        //// iii: Angle Offset

        public static Vector3 DirectionOffset(RunControl runcontrol)
        {            
            // Adjust yaw and pitch relative to the camera-to-vehicle directional vector 
            // Assume: Roll is level
            // Note: Radian conversion
            
            // [a] Yaw

            float h_fov = runcontrol.camera.FieldOfView;
            float yaw_offset = Other.PosNeg(runcontrol) * 0.5f * h_fov * (float)runcontrol.random.NextDouble() / 180f;

            // [b] Pitch

            Size screen = Game.ScreenResolution;
            float aspect_ratio = screen.Width / screen.Height;

            float v_fov = (float)(2 * Math.Atan(Math.Tan(h_fov / 2) * aspect_ratio));
            float pitch_offset = Other.PosNeg(runcontrol) * 0.5f * v_fov * (float)runcontrol.random.NextDouble() / 180f;

            // [c] Direction offset

            Vector3 directionoffset = new Vector3((float)(Math.Cos(pitch_offset) * Math.Cos(yaw_offset)), 
                                                  (float)(Math.Cos(pitch_offset) * Math.Sin(yaw_offset)), 
                                                  (float)(-Math.Sin(pitch_offset)));

            return directionoffset;
        }

        /// B: Take picture 

        public static void TakePicture(TestControl testcontrol, IDictionary<int, string> damage = null)
        {
            string filename = GenerateFileName(testcontrol, damage);                     // [a] Generate file name

            UI.Notify(filename); // ADJUST

            WriteToTiff.RobustBytesToTiff(filename);                                     // [b] Take picture                                        
        }

        public static string GenerateFileName(TestControl testcontrol, IDictionary<int, string> damage = null)
        {   
            List<string> filename_list = new List<string>
            {
                testcontrol.id.ToString(),                                             // [a] TestControl ID
                ((int)(testcontrol.altitude)).ToString(),                              // [b] Camera altitude
                testcontrol.isocclusion,                                               // [c] Check: Weather occlusion is present
                testcontrol.timeofday                                                  // [d] Check: Day or night
            };

            if (!(damage is null))
            {
                filename_list.Add(damage.Keys.ElementAt<int>(0).ToString());           // [e] Add damaged vehicle's id
                filename_list.Add(damage.Values.ElementAt<string>(0));                 // [f] Add damage capture position (if applicable)
            }

            string filename = String.Join("_", filename_list);                         // Note: w/o .tif
            return filename;
        }

        /// C: Take picture sequence of damaged vehicles
        public static void CaptureDamagedVehicle(RunControl runcontrol, TestControl testcontrol, Target target)
        {
            IDictionary<string, Vector3> cam_offsets = CollisionPositional(runcontrol, target.vehicle);     // [a] Camera positions

            foreach (KeyValuePair<string, Vector3> cam_offset in cam_offsets)                               // <car pos name, car pos vector3> 
            {
                target.vehicle.Speed = 0f;                                                                  // [b] Stop damaged vehicle
                TestControl.SetPlayerIntoVehicle(target.vehicle);                                           // [c] Place character in vehicle
                runcontrol.camera.AttachTo(target.vehicle, cam_offset.Value);                               // [d] Place camera at offset position

                Vector3 directionoffset = DirectionOffset(runcontrol);                                      // [e] Randomise camera yaw and and pitch
                runcontrol.camera.PointAt(target.vehicle, directionoffset);
                Script.Wait(100);
                
                WriteToTiff.PrepareGameBuffer(true);

                IDictionary<int, string> damage = new Dictionary<int, string>() { { target.damage.id, cam_offset.Key } };

                UI.Notify(damage.Values.ElementAt<string>(0));        // ADJUST

                TakePicture(testcontrol, damage);                                                           // [f] Take picture
                
                WriteToTiff.PrepareGameBuffer(false);               
            }
        }

        public static Vector2 COL_HEIGHT_R = new Vector2(0f, 2f);
        public static Vector2 COL_DIST_FR = new Vector2(6f, 10f);
        public static Vector2 COL_DIST_RR = new Vector2(5f, 8f);

        public static IDictionary<string, Vector3> CollisionPositional(RunControl runcontrol, Vehicle vehicle)
        {
            /// A: Directional unit vectors

            Vector3 forwardvector = vehicle.ForwardVector;
            Vector3 rightvector = vehicle.RightVector;
            Vector3 upvector = vehicle.UpVector;

            Vector3 offvector = 5 * upvector;

            /// B: Pertubations

            IEnumerable<List<Vector3>> pertubations
                = from no in Enumerable.Range(0, 4)
                  select new List<Vector3>() {
                      ((float)(COL_DIST_RR[0] + (COL_DIST_RR[1] - COL_DIST_RR[0]) * runcontrol.random.NextDouble()))*rightvector,
                      ((float)(COL_DIST_FR[0] + (COL_DIST_FR[1] - COL_DIST_FR[0]) * runcontrol.random.NextDouble()))*forwardvector,
                      ((float)(COL_HEIGHT_R[0] + (COL_HEIGHT_R[1] - COL_HEIGHT_R[0]) * runcontrol.random.NextDouble()))*upvector
                  };

            /// C: Positions

            List<Vector3> fr = pertubations.ElementAt(0);
            List<Vector3> fl = pertubations.ElementAt(1);
            List<Vector3> bl = pertubations.ElementAt(2);
            List<Vector3> br = pertubations.ElementAt(3);

            IDictionary<string, Vector3> offsets = new Dictionary<string, Vector3>()
            {
                {"frontright", offvector + fr[0] + fr[1] + fr[2]},
                {"frontleft", offvector - fr[0] + fr[1] + fr[2]},
                {"backleft", offvector - fr[0] - fr[1] + fr[2]},
                {"backright", offvector + fr[0] - fr[1] + fr[2]}
            };

            return offsets;
        }
    }
}
