using System;
using System.Data.SqlClient;

using GTA;
using GTA.Math;
using GTA.Native;

namespace DRS
{
    public class TestControl
    {
        // 1: Properties =========================================================================

        public int id;                            

        /// A: Test

        public DateTime teststarttime;
        public DateTime testendtime;
        public int testduration;

        /// B: Camera

        public Vector3 position;
        public Vector3 rotation;
        public int fov;
        public float height_above_ground;        

        /// C: Graphics

        public Matrix world_matrix;
        public Matrix view_matrix;
        public Matrix project_matrix;

        /// D: Accident

        public double speedofimpact;
        public double angleofimpact;

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

        // 3: Update ============================================================================

        public void Update()
        {
            testendtime = DateTime.Now;
            testduration = (testendtime - teststarttime).Duration().Milliseconds;
        }

        // 4: Database ==========================================================================

        public static string[] db_test_control_parameters =
        {
            "TestControlID", 
            "Duration"
        };

        public static string sql_testcontrol = DB.SQLCommand("TestControl", db_test_control_parameters);

        public void ToTestControl()
        {
            SqlConnection cnn = DB.InitialiseCNN();

            using (SqlCommand cmd = new SqlCommand(sql_testcontrol, cnn))
            {
                cmd.Parameters.AddWithValue("@TestControlID", id);
                cmd.Parameters.AddWithValue("@Duration", testduration);

                /// Overall: Write to database

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }

    }
}

