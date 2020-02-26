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
        public static void CaptureVicinity(RunControl runcontrol, TestControl testcontrol)
        {
            PositionWideCam(runcontrol, testcontrol);
        }

        public static void CaptureDamagedVehicle(RunControl runcontrol, Vehicle vehicle)
        {

        }

        // 1: Functions =================================================================================================

        /// A: Attach camera at random distance and angle to target vehicle

        // public static Vector2 CAM_OFF_R = new Vector2(10f, 100f);

        public static void PositionWideCam(RunControl runcontrol, TestControl testcontrol)
        {
            Game.Player.Character.SetIntoVehicle(testcontrol.target_vehicle, VehicleSeat.Any); // [b] Position player into vehicle
            
            //// i: Location offset

            Vector3 locationoffset = LocationOffset(runcontrol, testcontrol.target_vehicle);
            runcontrol.camera.AttachTo(testcontrol.target_vehicle, locationoffset);
            RunControl.RenderCreatedCameras(true);

            //// ii: Angle offset

            Script.Wait(1000);
        }

        //// ii: Location Offset 

        public static float CAM_MIN_DIST = 10f;                                                    // [a] In all positive upvector space
        public static float CAM_EXC_DIST = 90f;
        public static Vector3 LocationOffset(RunControl runcontrol, Vehicle vehicle)
        {
            Vector3 locationoffset =
                Other.PosNeg(runcontrol) * (CAM_MIN_DIST + (float)runcontrol.random.NextDouble() * CAM_EXC_DIST) * vehicle.RightVector +
                Other.PosNeg(runcontrol) * (CAM_MIN_DIST + (float)runcontrol.random.NextDouble() * CAM_EXC_DIST) * vehicle.ForwardVector +
                (CAM_MIN_DIST + (float)runcontrol.random.NextDouble() * CAM_EXC_DIST) * vehicle.UpVector;

            return locationoffset;
        }

        //// iii: Angle Offset

        public static void RotationOffset(RunControl runcontrol, Vehicle vehicle)
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

            float v_fov = (float)(2 * Math.Atan(Math.Tan(h_fov / 2) * aspect_ratio);
            float pitch_offset = Other.PosNeg(runcontrol) * 0.5f * v_fov * (float)runcontrol.random.NextDouble() / 180f;

            // [c] Direction offset

            Vector3 directionoffset = new Vector3((float)(Math.Cos(pitch_offset) * Math.Cos(yaw_offset)), 
                                                  (float)(Math.Cos(pitch_offset) * Math.Sin(yaw_offset)), 
                                                  (float)(-Math.Sin(pitch_offset)));

            runcontrol.camera.PointAt(vehicle, directionoffset);
        }

        

        


        // 5: Methods =========================================================================================










        public static void PictureSequence(RunControl runcontrol, TestControl testcontrol, Target target)
        {
            //// i: Create test control image directory

            string folder = @"D:\ImageDB\Car";                                             // [a] Type folder
            string testpath = Path.Combine(folder, testcontrol.id.ToString());             // [b] Test subfolder

            Directory.CreateDirectory(testpath);                                           // [c] Create directory

            //// iii: Take picture series

            IDictionary<string, Vector3> Positional = AerialConfiguration.CollisionPositional(runcontrol, target);
            IList<string> aerialkeys = Positional.Keys.ToList<string>();

            foreach (string aerialkey in aerialkeys)
            {
                //// > Create angle directory

                string anglepath = Path.Combine(testpath, aerialkey);
                Directory.CreateDirectory(anglepath);

                //// > Position camera at position and point at car center

                Vector3 offset = Positional[aerialkey];                                      // [a] Aerial position

                runcontrol.camera.AttachTo(target.vehicle, offset);                          // [b] Set camera position
                runcontrol.camera.PointAt(target.vehicle);                                   // [c] Point camera at vehicle           

                //// > Delay camera (allow the environment to load) and take picture

                Script.Wait(2000);                                                           // [d] Wait specified time to move camera
                TakePicture(runcontrol, testcontrol, anglepath, target.vehicle);             // [e] Take picture
            }
        }

        /// B: Take picture

        public static void TakePicture(RunControl runcontrol, TestControl testcontrol, string anglepath, Vehicle vehicle)
        {
            WriteToTiff.PrepareGameBuffer(true);                                             // [a] Prepare game buffer

            //// i: File name

            object[] filename_object = new object[4]
                {
                    "Car",                                                                 // [a] Test type
                    runcontrol.id.ToString(),                                              // [b] RunControl ID
                    testcontrol.id.ToString(),                                             // [c] TestControl ID
                    testcontrol.iscollision                                                // [d] Whether vehicle is onscreen
                };

            string filename = String.Join("_", filename_object);                           // Note: w/o .tif
            string filepath = Path.Combine(anglepath, filename);

            //// iii: Take Picture 

            WriteToTiff.RobustByteToTiff(filepath);                                                    // [a] Take picture
            WriteToTiff.PrepareGameBuffer(false);                                             // [b] Return game to normal state
        }

        /*
public static void Cause(RunControl runcontrol, TestControl testcontrol, Environment environment)
{
    //// ii: Position camera and take picture

    AttachCamToCar(runcontrol, testcontrol, vehicle, 20, 8f, 8f);                          /// [a] Move player and camera to vehicle location (with offset)

    /// D: Response

    TakePicture(runcontrol, testcontrol, vehicle);

    /// E: Remove vehicle persistence

    Function.Call(Hash.RENDER_SCRIPT_CAMS, 0, 1, 0, 0, 0);
    Script.Wait(1);

    Game.Player.Character.IsVisible = false;
}
*/



        /// E: Image paths

        /* 

        public static string[] ImagePaths(RunControl runcontrol, TestControl testcontrol, string testpath, string pos, string atfault = null)
        {
            //// i: Create position directory

            string pospath = Path.Combine(testpath, pos);
            Directory.CreateDirectory(pospath);

            //// ii: Create file name UPTO directory

            object[] filenamepos_object;


            filenamepos_object = new object[4]
            {
                runcontrol.type,                                             // [a] Test type
                runcontrol.id.ToString(),                                    // [b] RunControl ID
                testcontrol.id.ToString(),                                   // [c] TestControl ID
                pos                                                          // [d] Camera aerial position
            };

            string filenamepos_string = String.Join("_", filenamepos_object);

            //// ii: Specify image paths

            string[] imagepaths = new string[3];

            int i = 0;
            foreach(string imagetype in Other.imagetypes)
            {
                string filename = filenamepos_string + "_" + imagetype + ".tif";    // [f] Formulate file name
                imagepaths[i] = Path.Combine(pospath, filename);                    // [g] Amalgamate complete image path
                i++;                                                                // [h] Iterator
            }

            return imagepaths;
        }
        */

        // TYPE C: COLLISION POSITIONAL 

        public static Vector2 COL_HEIGHT_R = new Vector2(0f, 2f);
        public static Vector2 COL_DIST_FR = new Vector2(6f, 10f);
        public static Vector2 COL_DIST_RR = new Vector2(5f, 8f);

        public static IDictionary<string, Vector3> CollisionPositional(RunControl runcontrol, Target target)
        {
            /// A: Directional unit vectors

            Vector3 forwardvector = target.vehicle.ForwardVector;
            Vector3 rightvector = target.vehicle.RightVector;
            Vector3 upvector = target.vehicle.UpVector;

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
