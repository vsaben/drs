using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.IO;

using GTA;
using GTA.Math;
using GTA.Native;

namespace DRS
{
    public class TestControl
    {
        public static string OUTPUT_PATH = "D:\\ImageDB\\Surveillance\\";    // Image, JSON output path 

        public int id;
        public string base_filename;                                         // File name w/o specification and ext
        public Vector3 baseposition;                                         // Drone base position 

        /// A: Test time

        public DateTime teststarttime;
        public DateTime testendtime;
        public int testduration;

        /// B: Entities

        public List<Entity> frozen_entities; 
        public EntitiesInLOS entities_wide;                                  // Entities: Vehicles, damaged, peds [wide frame]

        /// C: Collision

        public Vehicle target_vehicle;
        public Vehicle colliding_vehicle;

        /// D: Camera - Damage Control 

        public DamagedInstance damaged_instance;
        public bool iswide;                                                 // Wide or near camera angle
        public bool iswidecaptured;                                         // Wide view captured

        // 1: Setup and update =========================================================================================

        public static TestControl Setup()
        {
            // Function - Output: Setup and output TestControl instance

            RunControl.TurnOffGameplayOptions(true);

            TestControl testcontrol = new TestControl()
            {
                id = DB.LastID("TestControl") + 1,              // [a] Get next test ID from SQL database     
                teststarttime = DateTime.Now,                   // [b] Record test start time
                damaged_instance = DamagedInstance.Setup(),     // [c] Current damage id [near capture]
                iswide = true,                                  // [d] Wide or near camera angle 
                iswidecaptured = false                          // [e] Check if wide view captured
            };

            testcontrol.GenerateBaseFilePath();                 // [d] Generate base file path: Output path + 6-digit ID 

            return testcontrol;
        }

        public void TestUpdate()
        {   
            // Function: Update time, remove damaged or involved vehicles
            // Output: Send test info to SQL DB

            testendtime = DateTime.Now;
            testduration = (testendtime - teststarttime).Duration().Milliseconds;
            ToDB();           
        }

        public static void DeleteDamagedVehicles(RunControl runcontrol)
        {
            // Function - Output: Deletes damaged (in and out of LOS) vehicles

            World.GetNearbyVehicles(runcontrol.camera.Position, 1000f)
                .Where(x => Damage.DamageCheck(x)).ToList().ForEach(x => x.Delete());
        }

        // 3: File name ============================================================================================
        
        public void GenerateBaseFilePath() => base_filename = OUTPUT_PATH + IntToIDString(6, id);

        public static string IntToIDString(int length, int number)
        {
            int num_char = number.ToString().Length;
            string res_string = string.Concat(Enumerable.Repeat("0", length - num_char)) + number.ToString();
            return res_string;
        }

        // 4: Save test information to SQL ============================================================

        public static string[] db_test_control_parameters =
        {
            "TestControlID",
            "Duration",
            "NumVehicles",
            "NumDamagedVehicles",
            "NumPeds"
        };

        public static string sql_testcontrol = DB.SQLCommand("TestControl", db_test_control_parameters);

        public void ToDB()
        {
            SqlConnection cnn = DB.InitialiseCNN();

            using (SqlCommand cmd = new SqlCommand(sql_testcontrol, cnn))
            {
                cmd.Parameters.AddWithValue("@TestControlID", id);
                cmd.Parameters.AddWithValue("@Duration", testduration);
                cmd.Parameters.AddWithValue("@NumVehicles", entities_wide.vehicles.Count);
                cmd.Parameters.AddWithValue("@NumDamagedVehicles", entities_wide.damaged_vehicles.Count);
                cmd.Parameters.AddWithValue("@NumPeds", entities_wide.peds.Count);

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }
    }

    public class EntitiesInLOS
    {
        public List<Vehicle> vehicles;
        public List<Vehicle> damaged_vehicles;
        public List<Target> targets;
        public List<PedSummary> peds;
        public static EntitiesInLOS Setup(RunControl runcontrol)
        {
            EntitiesInLOS inlos = new EntitiesInLOS() { };
            inlos.Update(runcontrol);
            return inlos;           
        }
        public void Update(RunControl runcontrol)
        {
            VehiclesInLOS(runcontrol);
            TargetsDamagedInLOS();
            PedsInLOS(runcontrol);     
        }

        public void VehiclesInLOS(RunControl runcontrol)
        {
            vehicles = World.GetNearbyVehicles(runcontrol.camera.Position, 300f).Where(x => IsInLOS(x)).ToList();
        }

        public void TargetsDamagedInLOS()
        {            
            damaged_vehicles = new List<Vehicle>() { };
            targets = new List<Target>() { };

            if (vehicles.Count == 0) return;

            int dam_counter = 0;

            foreach (Vehicle vehicle in vehicles)
            {
                bool dam_check = Damage.DamageCheck(vehicle);
                int dam_id = 0;

                if (dam_check)
                {
                    dam_counter += 1;
                    dam_id = dam_counter;
                    damaged_vehicles.Add(vehicle);
                }

                Target target = Target.Setup(vehicle, dam_id);
                targets.Add(target);
            }             
        }
        public void PedsInLOS(RunControl runcontrol)
        {
            List<Ped> allpeds = World.GetNearbyPeds(runcontrol.camera.Position, 300f).Where(x => IsInLOS(x)).ToList();
            peds = allpeds.Select(x => PedSummary.Setup(x)).ToList();
        }
        public static bool IsInLOS(Entity entity) => entity.IsOnScreen && !entity.IsOccluded && entity.IsVisible;
    }

    public class DamagedInstance 
    {
        public int id;                      // Damaged instance's ID
        public DamagePosition dam_pos;      // Damage position
        public static DamagedInstance Setup()
        {
            DamagedInstance instance = new DamagedInstance()
            {
                id = 0
            };

            return instance;
        }
    }
}

