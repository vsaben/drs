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
        // 1: Camera Control ==================================================================================================================
        public static bool PositionCam(RunControl runcontrol, TestControl testcontrol, double theta = 0)
        {
            // Function: Attach camera at random distance and angle to target vehicle, maintaining LOS (3 tries)
            // Output: Bool - Whether target vehicle is in LOS

            for (int i = 0; i < 3; i++)
            {                
                LocationOffset(runcontrol, testcontrol, theta);                 // [a] Position created camera (location only)    
                Script.Wait(2000);
                
                bool inlos = EntitiesInLOS.IsInLOS(testcontrol.target_vehicle); // [b] Check if the target vehicle is in LOS
                if (inlos)
                {                   
                    RotationOffset(runcontrol, testcontrol);                    // [c] Adjust camera angle                                 
                    Script.Wait(2000);
                    return inlos;                                               // [d] Return true if target vehicle is in LOS    
                }
            }

            return false;                                                       // [f] Return false if target vehicle not in LOS (after 3 attempts)
        }

        /// A: Location Offset

        public static Dictionary<string, Vector2> LOCATION_OFFSET_PARAMS_WIDE = new Dictionary<string, Vector2>()
        {
            {"H",  new Vector2(10f, 50f)},                                     // H = Height
            {"W",  new Vector2(0f, 50f)},                                      // W = Width
            {"AZI", new Vector2(1/6f, 3/6f)}                                   // AZI = Azimuth pi factor                                                           
        };

        public static Dictionary<string, Vector2> LOCATION_OFFSET_PARAMS_NEAR = new Dictionary<string, Vector2>()
        {
            {"H", new Vector2(5f, 15f)},                                       
            {"W",  new Vector2(5f, 10f)},                                      
            {"AZI", new Vector2(1/6f, 2.5f/6f)}                                                                                        
        };

        public static void LocationOffset(RunControl runcontrol, TestControl testcontrol, double theta = 0)
        {
            // Function: Determine camera position offset to target vehicle
            // Output: Move camera position

            /// A: Select correct offset parameters

            Dictionary<string, Vector2> location_offset_params = testcontrol.iswide ? 
                LOCATION_OFFSET_PARAMS_WIDE : LOCATION_OFFSET_PARAMS_NEAR;
            Vector2 h = location_offset_params["H"];
            Vector2 w = location_offset_params["W"];
            Vector2 azi = location_offset_params["AZI"];
            if (testcontrol.iswide) theta = 2 * Math.PI * (float)runcontrol.random.NextDouble();
            
            /// B: Determine location offset

            float H = h[0] + (h[1] - h[0]) * (float)runcontrol.random.NextDouble();                 // XY
            float W = w[0] + (w[1] - w[0]) * (float)runcontrol.random.NextDouble();                 // XY
            float azimuth = azi[0] + (azi[1] - azi[0]) * (float)runcontrol.random.NextDouble();     // XY-Z
            Vector3 location_offset = new Vector3(W * (float)Math.Sin(theta), W * (float)Math.Cos(theta), H * (float)Math.Cos(azimuth));
            
            /// C: Control camera
            
            runcontrol.camera.Position = testcontrol.target_vehicle.Position + location_offset;     // [a] Position camera
            RunControl.RenderCreatedCameras(true);                                                  // [b] Switch camera on 
            runcontrol.camera.PointAt(testcontrol.target_vehicle);                                  // [c] Point camera directly at the target vehicle 
        }

        public static Dictionary<DamagePosition, double> ThetaPositional(RunControl runcontrol)
        {
            // Function: Control offset angle in XY plane corresponding to 4 damage capture corners
            // Output: 4 theta angles

            Dictionary<DamagePosition, double> res_theta = new Dictionary<DamagePosition, double>(); 
                     
            foreach(DamagePosition position in Enum.GetValues(typeof(DamagePosition))) 
            {
                double theta_deg = (int)position * 90 + 15 + 60 * runcontrol.random.NextDouble();
                double theta_rad = theta_deg / 180 * Math.PI;

                res_theta.Add(position, theta_rad);
            }

            return res_theta;                                                      
        }

        /// B: Angle Offset

        public static Dictionary<string, float> FOV_MARGIN_PER = new Dictionary<string, float>() {
            {"Near", 0.4f },
            {"Far", 0.2f}
        };        
        public static void RotationOffset(RunControl runcontrol, TestControl testcontrol)
        {
            // Function: Adjust yaw and pitch relative to the camera-to-vehicle rotation (Assume roll is level) 
            //           whilst maintaining view of the target vehicle
            // Output: Adjust camera rotation 
            // Notes: 
            //        - Rotation: <pitch, roll, yaw>
            //        - Half-rotation about axis
            //        - Radian conversion 
            //        - FOV margin

            /// A: Field-of-view

            float vfov = runcontrol.camera.FieldOfView;
            Size screen = Game.ScreenResolution;

            float aspect_ratio = (float)screen.Width / screen.Height;
            float hfov = HFOV(vfov, aspect_ratio);                                            // [a] Calculate horizontal FOV

            /// B: Rotation matrix adjustment 

            float fov_margin_per = testcontrol.iswide ? FOV_MARGIN_PER["Far"] : FOV_MARGIN_PER["Near"]; 

            float yaw_offset = RandomFOVOffset(runcontrol, hfov, fov_margin_per);
            float pitch_offset = RandomFOVOffset(runcontrol, vfov, fov_margin_per);
            Vector3 rotationoffset = new Vector3(pitch_offset, 0f, yaw_offset);

            /// C: Adjust rotation matrix

            Vector3 camera_to_vehicle_rotation = runcontrol.camera.Rotation;                 // [b] Find direct camera to target rotation
            runcontrol.camera.StopPointing();                                                // [c] Stop camera pointing at the target
            runcontrol.camera.Rotation = camera_to_vehicle_rotation + rotationoffset;        // [d] Adjust camera rotation with offset
        }
        public static float HFOV(float vfov, float aspect)
        {
            // Function - Output: Calculate horizontal FOV from vertical FOV and the aspect ratio (in degrees)

            float vfov_rad = (float)((vfov / 180) * Math.PI);
            float hfov_rad = (float)(2 * Math.Atan(Math.Tan(vfov_rad / 2) * aspect));
            float hfov = (float)((hfov_rad / Math.PI) * 180);
            return hfov;
        }
        public static float RandomFOVOffset(RunControl runcontrol, float fov, float fov_margin_per)
        {
            // Function - Output: Calculate random fov offset (uniform) maintaining the original focal point

            float half_offset = 0.5f * fov * (1 - fov_margin_per);
            float res_offset = half_offset * (-1 + 2 * (float)runcontrol.random.NextDouble());
            return res_offset;
        }

        // 2: Take picture =======================================================================================================       
        public static void TakePicture(TestControl testcontrol)
        {
            // Function: Take picture of current screen
            // Output: Colour, depth and stencil images

            string filepath = ST.GenerateFilePath(testcontrol);                       // [a] Generate file path w/o ext.
            WriteToTiff.RobustBytesToTiff(filepath);                                  // [b] Take picture                                        
        }

        public static void Capture(RunControl runcontrol, TestControl testcontrol, Environment environment, double theta = 0)
        {
            // Function: Prepare game buffer, take picture of screen and save scene information (if target inlos)
            // Output: Images and annotations

            bool inlos = PositionCam(runcontrol, testcontrol, theta);             // [a] Position camera at random location and directional offset to target             

            if (!inlos)
            {                
                UI.ShowSubtitle("Not in LOS", 1000);
                return;
            }

            WriteToTiff.PrepareGameBuffer(true);                                      
            TakePicture(testcontrol);                                             // [b] Take a picture            
            ST.Save(runcontrol, testcontrol, environment);                        // [c] Extract annotations: Camera, control, environment, target, ped (JSON)
            WriteToTiff.PrepareGameBuffer(false);

            if (testcontrol.iswide) testcontrol.iswidecaptured = true;
        }
        
        // 3: Take picture sequence of damaged vehicles ===============================================================================================
        
        public static void CaptureDamagedVehicles(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {
            // Function: Captures 4 image sequence of all damaged instances from wide view
            // Output: 4 x image angles, 4 x json

            testcontrol.iswide = false;
            if (testcontrol.entities_wide.damaged_vehicles.Count == 0) return;

            List<Vehicle> allowed_damaged_vehicles = testcontrol.entities_wide.damaged_vehicles
                .Where(x => VehicleSelection.ALLOWED_VEHICLE_CLASSES.Contains(x.ClassType)).ToList();
            if (allowed_damaged_vehicles.Count == 0) return;

            foreach (Vehicle damaged_vehicle in allowed_damaged_vehicles)                       // [a] Only capture damaged vehicles [allowed classes]
            {
                testcontrol.target_vehicle = damaged_vehicle;                                   // [b] Set damaged vehicle as target vehicle
                testcontrol.damaged_instance.id += 1;                                           // [c] Iterate damage counter (for identifier/file name purposes)

                CaptureDamagedVehicle(runcontrol, testcontrol, environment);                    // [d] Capture damaged vehicle instance
            }
        }       
        public static void CaptureDamagedVehicle(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {
            Dictionary<DamagePosition, double> thetas = ThetaPositional(runcontrol);            // [a] Calculate random theta values in XY quadrants (in radians)

            foreach (DamagePosition position in Enum.GetValues(typeof(DamagePosition)))
            {
                testcontrol.damaged_instance.dam_pos = position;                                // [b] Change damage corner
                Capture(runcontrol, testcontrol, environment, thetas[position]);                // [c] Capture damaged vehicle instance from corner
            }            
        }
    }
}
