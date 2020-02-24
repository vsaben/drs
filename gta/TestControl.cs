using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;

using GTA;
using GTA.Math;
using GTA.Native;
using GTAVisionUtils;

using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

using SharpDX.Mathematics;

namespace DRS
{
    public class TestControl
    {
        // 1: Properties =========================================================================

        public int id;                            

        /// A: Test time

        public DateTime teststarttime;
        public DateTime testendtime;
        public int testduration;

        /// B: Camera

        public Vector3 location;
        public Vector3 rotation;
        public float fov;                     // Camera field-of-view

        /// C: Graphics

        public DenseMatrix W;                 // World Matrix
        public DenseMatrix V;                 // View Matrix
        public DenseMatrix P;                 // Projection Matrix

        /// D: Entities

        public Vehicle[] vehicles_in_los;     // Vehicle in line-of-sight
        public Ped[] peds_in_los;             // Pedestrians in line-of-sight

        /// E: Collision

        public Vehicle target_vehicle;
        public Vehicle colliding_vehicle;

        public float angleofimpact;
        public float initialspeed;

        // 2: Setup =============================================================================
        
        public static TestControl Setup()
        {
            TestControl testcontrol = new TestControl()
            {
                id = DB.LastID("TestControl") + 1,
                teststarttime = DateTime.Now 
            };
                       
            return testcontrol;
        }

        public void Update(RunControl runcontrol)
        {
            // A: Test

            testendtime = DateTime.Now;
            testduration = (testendtime - teststarttime).Duration().Milliseconds;

            // B: Camera

            location = runcontrol.camera.Position;
            rotation = runcontrol.camera.Rotation;
            fov = runcontrol.camera.FieldOfView;

            // C: Graphics

            List<DenseMatrix> list_wvp = GetWVP();
            W = list_wvp.ElementAt<DenseMatrix>(0);
            V = list_wvp.ElementAt<DenseMatrix>(1);
            P = list_wvp.ElementAt<DenseMatrix>(2);

            // D: Entities

            vehicles_in_los = VehiclesInLOS();
            peds_in_los = PedsInLOS();
        }

        // 3: Functions =========================================================================

        public static Vehicle[] VehiclesInLOS()
        {
            return (Vehicle[])World.GetAllVehicles().Where<Vehicle>(x => LOS(x));
        }

        public static Ped[] PedsInLOS()
        {
            return (Ped[])World.GetAllPeds().Where<Ped>(x => LOS(x));
        }

        public static bool LOS(Entity entity)
        {
            return entity.IsOnScreen && !entity.IsOccluded;
        }

        public static List<DenseMatrix> GetWVP()
        {
            /// A: VisionNative Matrices

            rage_matrices? constants = VisionNative.GetConstants();

            DenseMatrix W = (DenseMatrix)MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.
                OfColumnMajor(4, 4, constants.Value.world.ToArray()).ToDouble();
            DenseMatrix WV = (DenseMatrix)MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.
                OfColumnMajor(4, 4, constants.Value.worldView.ToArray()).ToDouble();
            DenseMatrix WVP = (DenseMatrix)MathNet.Numerics.LinearAlgebra.Single.DenseMatrix.
                OfColumnMajor(4, 4, constants.Value.worldViewProjection.ToArray()).ToDouble();

            /// B: World-View-Projection Separation

            DenseMatrix V = (DenseMatrix)W.Inverse() * WV;
            DenseMatrix P = (DenseMatrix)WV.Inverse() * WVP;

            List<DenseMatrix> res_wvp = new List<DenseMatrix>(3) { W, V, P };
            return res_wvp;
        }

        // 4: Database ==========================================================================

        public static string[] db_test_control_parameters =
        {
            "TestControlID", 
            "Duration",
            "NumVehicles",
            "NumPeds"
        };

        public static string sql_testcontrol = DB.SQLCommand("TestControl", db_test_control_parameters);

        public void ToTestControl()
        {
            SqlConnection cnn = DB.InitialiseCNN();

            using (SqlCommand cmd = new SqlCommand(sql_testcontrol, cnn))
            {
                cmd.Parameters.AddWithValue("@TestControlID", id);
                cmd.Parameters.AddWithValue("@Duration", testduration);
                cmd.Parameters.AddWithValue("@NumVehicles", vehicles_in_los.Length);
                cmd.Parameters.AddWithValue("@NumPeds", peds_in_los.Length);

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }
    }
}

