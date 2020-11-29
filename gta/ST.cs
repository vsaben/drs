using System;
using System.Drawing;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

using GTA;
using GTA.Math;
using GTAVisionUtils;

using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;

namespace DRS
{
    public class ST
    {
        public CameraST camerast;
        public ControlST controlst;
        public Environment environmentst;
        public List<Target> targetst;
        public List<PedSummary> pedst;

        // 1: Save information to JSON ======================================================================

        public static void Save(RunControl runcontrol, TestControl testcontrol, Environment environment)
        {
            // Function - Output: Save entity, camera, environment and control information to JSON files

            EntitiesInLOS entitiesinlos = EntitiesInLOS.Setup(runcontrol);
            if (testcontrol.iswide) testcontrol.entities_wide = entitiesinlos;

            environment.Update();

            ST st = new ST()
            {
                camerast = CameraST.Setup(runcontrol),
                controlst = ControlST.Setup(runcontrol, testcontrol, entitiesinlos, environment),
                environmentst = environment,
                targetst = entitiesinlos.targets,
                pedst = entitiesinlos.peds
            };

            ToJSON(testcontrol, st);
        }

        public static Dictionary<DamagePosition, string> PosNameDict = new Dictionary<DamagePosition, string>()
        {
            { DamagePosition.frontright, "FR" },
            { DamagePosition.backright, "BR" },
            { DamagePosition.backleft, "BL" },
            { DamagePosition.frontleft, "FL" }
        };

        public static string GenerateFilePath(TestControl testcontrol, bool isjson = false)
        {
            // Function - Output: Generate file path for wide- and near-frame angles

            string ext = isjson ? ".json" : "";

            string specification = testcontrol.iswide ? "W00RD" :
                "D" + TestControl.IntToIDString(2, testcontrol.damaged_instance.id) + PosNameDict[testcontrol.damaged_instance.dam_pos];

            string filepath = testcontrol.basepath + specification + ext;
            return filepath;
        }
        public static void ToJSON(TestControl testcontrol, ST st)
        {
            // Function - Output: Save ST class to JSON file

            string path = GenerateFilePath(testcontrol, true);

            using (StreamWriter file = File.CreateText(path))
            {
                JsonSerializer serializer = new JsonSerializer()
                {
                    Formatting = Formatting.None,
                    PreserveReferencesHandling = PreserveReferencesHandling.None,
                    ReferenceLoopHandling = ReferenceLoopHandling.Ignore
                };

                serializer.Serialize(file, st);
            }
        }

        // 2: ST Classes ===========================================================================================

        public class CameraST
        {
            public List<float> location;
            public List<float> rotation;            
            public float vfov;
            public float fclip;
            public float nclip;
            public int screenW;
            public int screenH;
            public Matrix<double> C;
            public Matrix<double> V;
            public Matrix<double> P;

            public static CameraST Setup(RunControl runcontrol)
            {
                Camera camera = runcontrol.camera;
                Size screenres = Game.ScreenResolution;
                Dictionary <string, Matrix<double>> matrices = GetVP();

                CameraST camerast = new CameraST()
                {
                    location = camera.Position.VecToList(),
                    rotation = camera.Rotation.VecToList(),
                    vfov = camera.FieldOfView,
                    fclip = camera.FarClip,
                    nclip = camera.NearClip,
                    screenW = screenres.Width,
                    screenH = screenres.Height,
                    C = matrices["C"],
                    V = matrices["V"],
                    P = matrices["P"]
                };

                return camerast;
            }

            public static Dictionary<string, Matrix<double>> GetVP()
            {
                VisionNative.GetConstants(out rage_matrices constants);

                Matrix<double> W = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4, constants.world.ToArray()).ToDouble();
                Matrix<double> VW = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4, constants.worldView.ToArray()).ToDouble();
                Matrix<double> PVW = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4, constants.worldViewProjection.ToArray()).ToDouble();
                Matrix<double> C = MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.OfColumnMajor(4, 4, CameraHelper.GetCameraMatrix().ToArray()).ToDouble();
                
                Dictionary<string, Matrix<double>> res_vp = new Dictionary<string, Matrix<double>>()
                {
                  {"C", C },
                  {"V", VW * W.Inverse() },
                  {"P", PVW * VW.Inverse() }
                };

                return res_vp;
            }
        }

        public class ControlST
        {
            public Dictionary<string, int> ids;               // IDs: "run", "test"  
            public Dictionary<string, int> nums;              // Nums: "vehicles", "damaged_vehicles", "pedestrians"
            public Dictionary<string, string> assess_factors; // Factors: "occlusion", "timeofday", "altitude" 

            public static ControlST Setup(RunControl runcontrol, TestControl testcontrol, EntitiesInLOS entitiesinlos, Environment environment)
            {
                ControlST controlst = new ControlST()
                {
                    ids = new Dictionary<string, int>()
                    {
                        { "run", runcontrol.id },
                        { "test", testcontrol.id }
                    },

                    nums = new Dictionary<string, int>()
                    {
                        {"vehicles", entitiesinlos.vehicles.Count},
                        {"vehicles_damaged", entitiesinlos.damaged_vehicles.Count},
                        {"peds", entitiesinlos.peds.Count}
                    },

                    assess_factors = new Dictionary<string, string>()
                    {
                        {"occlusion", OcclusionCheck(environment)},
                        {"timeofday", DayCheck(environment)},
                        {"altitude", GetAltitude(runcontrol.camera).ToString()}
                    },
                };

                return controlst;
            }

            public static int GetAltitude(Camera camera) => (int)(camera.Position.Z - World.GetGroundHeight(camera.Position));

            public static string DayCheck(Environment environment)
            {
                TimeSpan DAYSTART = new TimeSpan(7, 0, 0);
                TimeSpan NIGHTSTART = new TimeSpan(19, 0, 0);

                return (environment.gametime >= DAYSTART && environment.gametime < NIGHTSTART) ? "d" : "n";
            }
            public static string OcclusionCheck(Environment environment) => (environment.rainlevel > 0 | environment.snowlevel > 0) ? "oc" : "noc";
        }

    }
}


    

