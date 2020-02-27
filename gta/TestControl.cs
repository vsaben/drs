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
        public float altitude;

        /// C: Graphics

        public DenseMatrix W;                 // World Matrix
        public DenseMatrix V;                 // View Matrix
        public DenseMatrix P;                 // Projection Matrix

        /// D: Entities

        public Vehicle[] vehicles_in_los;     // Vehicle in line-of-sight
        public Ped[] peds_in_los;             // Pedestrians in line-of-sight

        public int numvehicles;               // Number of vehicles in line-of-sight
        public int numdamaged;                // Number of damaged vehicles

        /// E: Collision

        public Vehicle target_vehicle;
        public Vehicle colliding_vehicle;

        public float angleofimpact;
        public float initialspeed;

        /// F: Separate assessment factors

        public string isocclusion;
        public string timeofday;

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

        public void Update(RunControl runcontrol, Environment environment)
        {
            // A: Test

            testendtime = DateTime.Now;
            testduration = (testendtime - teststarttime).Duration().Milliseconds;

            // B: Camera

            location = runcontrol.camera.Position;
            rotation = runcontrol.camera.Rotation;
            fov = runcontrol.camera.FieldOfView;
            altitude = GetAltitude(runcontrol);

            // C: Graphics

            List<DenseMatrix> list_wvp = GetWVP();
            W = list_wvp.ElementAt<DenseMatrix>(0);
            V = list_wvp.ElementAt<DenseMatrix>(1);
            P = list_wvp.ElementAt<DenseMatrix>(2);

            // D: Entities

            vehicles_in_los = VehiclesInLOS();
            peds_in_los = PedsInLOS();

            // F: Separate assessment factors

            isocclusion = OcclusionCheck(environment); 
            timeofday = DayCheck(environment);
        }

        // 3: Functions =========================================================================

        public static Vehicle[] VehiclesInLOS()
        {
            return World.GetAllVehicles().Where<Vehicle>(x => LOS(x)).ToArray();
        }

        public static Ped[] PedsInLOS()
        {
            return World.GetAllPeds().Where<Ped>(x => LOS(x)).ToArray();
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

        public static void SetPersistence(List<Entity> entities, bool persist)
        {
            if (persist)
            {
                foreach (Entity entity in entities) entity.IsPersistent = true;
            }
            else
            {
                foreach (Entity entity in entities) entity.IsPersistent = false;
            }
        }
        public static int GetAltitude(RunControl runcontrol)
        {
            return (int)(runcontrol.camera.Position.Z - World.GetGroundHeight(runcontrol.camera.Position));
        }
        public static string DayCheck(Environment environment)
        {
            TimeSpan DAYSTART = new TimeSpan(7, 0, 0);
            TimeSpan NIGHTSTART = new TimeSpan(19, 0, 0);

            return (environment.gametime >= DAYSTART && environment.gametime < NIGHTSTART) ? "d" : "n";
        }

        public static string OcclusionCheck(Environment environment)
        {
            return (environment.rainlevel > 0 | environment.snowlevel > 0) ? "oc" : "noc";
        }

        public static bool DamageCheck(Vehicle vehicle)
        {
            bool istyreburst = TireBurstCheck(vehicle);        // Unable to check for puncture        
            bool isbodydamaged = vehicle.BodyHealth < 1000f;

            return isbodydamaged | istyreburst;
        }
        public static bool TireBurstCheck(Vehicle vehicle)
        {
            IList<VehicleTyre> possibletyres = Target.PossibleTyres(vehicle);

            foreach (VehicleTyre tyre in possibletyres)
            {
                bool is_burst = vehicle.IsTireBurst((int)tyre);
                if(is_burst) return is_burst;
            }
            return false;
        }

        public static void CaptureVehicles(RunControl runcontrol, TestControl testcontrol)
        {
            testcontrol.numvehicles = testcontrol.vehicles_in_los.Length;

            Target target;

            foreach (Vehicle vehicle in testcontrol.vehicles_in_los)
            {
                bool istarget = object.ReferenceEquals(testcontrol.target_vehicle, vehicle);
                bool iscollider = object.ReferenceEquals(testcontrol.colliding_vehicle, vehicle);

                if (DamageCheck(vehicle))
                {
                    testcontrol.numdamaged += 1;
                    int damage_id = testcontrol.numdamaged;

                    UI.Notify(damage_id.ToString());

                    target = Target.Setup(vehicle, istarget, iscollider, damage_id);
                    Response.CaptureDamagedVehicle(runcontrol, testcontrol, target);
                    vehicle.Delete();
                }                
                else
                {
                    target = Target.Setup(vehicle, istarget, iscollider);
                }
                
                //// JSON FILE STORAGE/PLACEMENT
            }
        }

        // 4: Database ==========================================================================

        public static string[] db_test_control_parameters =
        {
            "TestControlID", 
            "Duration",
            "NumVehicles",
            "NumDamaged"
        };

        public static string sql_testcontrol = DB.SQLCommand("TestControl", db_test_control_parameters);

        public static void ToDB(TestControl testcontrol, SqlConnection cnn)
        {
            using (SqlCommand cmd = new SqlCommand(sql_testcontrol, cnn))
            {
                cmd.Parameters.AddWithValue("@TestControlID", testcontrol.id);
                cmd.Parameters.AddWithValue("@Duration", testcontrol.testduration);
                cmd.Parameters.AddWithValue("@NumVehicles", testcontrol.numvehicles);
                cmd.Parameters.AddWithValue("@NumPeds", testcontrol.numdamaged);

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }
    }
}

