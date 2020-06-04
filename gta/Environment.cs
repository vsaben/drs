using System;
using System.Linq;
using System.Collections.Generic;
using System.Data.SqlClient;

using GTA;
using GTA.Native;
using GTA.Math;

namespace DRS
{
    public class Environment
    {                    
        public TimeSpan gametime;               // In-game time
        public string weather;                  // Weather type
        public float rainlevel;                 // Level of rain
        public float snowlevel;                 // Level of snow
        
        // 1: Setup =======================================================================================

        public static Environment Setup(RunControl runcontrol, TestControl testcontrol)
        {
            // Function - Output: Setup and output Environment instance

            testcontrol.baseposition = SetRandomBasePosition(runcontrol);     // [a] Choose random drone base 

            Environment environment = new Environment
            {                
                weather = SetRandomWeather(runcontrol)                        // [b] Randomise weather                                                                           
            };

            Game.Player.Character.Position = testcontrol.baseposition;        // [c] Move player (and camera) to the new base
            SetRandomTime(runcontrol);                                        // [d] Randomise game time

            Script.Wait(3000);                                                // [e] Allow game to load

            return environment;
        }

        public void Update()
        {
            gametime = World.CurrentDayTime;
            rainlevel = GetRainLevel();
            snowlevel = GetSnowLevel();
        }

        // 2: Drone base locations ============================================================================

        public static IDictionary<string, Vector3> DroneBases = new Dictionary<string, Vector3>()
        {
            {"Airport - Helipad", new Vector3(-1102.290f, -2894.520f, 13.947f)},
            {"CLS Medical Center - Helipad", new Vector3(307.500f, -1458.930f, 46.510f)},
            {"Construction - Roof", new Vector3(-160.457f, -993.4858f, 254.1315f)},
            {"Country - Roof", new Vector3(2730.766f, 3468.552f, 73.70257f)},
            {"FIB - Roof", new Vector3(135.522f, -749.001f, 266.610f)},
            {"IAA - Roof", new Vector3(130.682f, -634.945f, 262.851f)},
            {"Kayton - Roof", new Vector3(-810.4823f, -609.8124f, 101.2702f)},
            {"Lombank - Helipad", new Vector3(-1584.083f, -569.4615f, 116.3276f)},
            {"Los Santos Customs - Roof", new Vector3(-327.3734f, -147.7118f, 63.73022f)},
            {"Low Roof 1" , new Vector3(-1031.616f, -2163.505f, 31.88425f)},
            {"Maibatsu Motors Inc. - Roof", new Vector3(895.561f, -987.624f, 44.271f)},
            {"Maze Bank - Helipad", new Vector3(-74.942f, -818.635f, 327.174f)},
            {"Maze Bank Arena - Roof", new Vector3(-324.300f, -1968.545f, 67.002f)},
            {"Mill - Roof", new Vector3(876.8009f, -1928.988f, 96.08694f)},
            {"Opposite Church - Roof", new Vector3(187.0198f, 230.8023f, 143.6602f)},
            {"Overlook Bridge - Roof", new Vector3(-846.7531f, -2142.469f, 101.3962f)},
            {"Paleto Bay Sheriff's Office - Helipad", new Vector3(-466.686f, 6013.817f, 32.000f)},
            {"Rancho Police - Helipad", new Vector3(369.430f, -1601.830f, 36.950f)},
            {"Rancho Police Parking - Roof", new Vector3(334.210f, -1644.770f, 98.496f)},
            {"Square - Roof", new Vector3(334.2882f, -1641.32f, 98.49608f)},
            {"Triangle - Roof", new Vector3(-999.7039f, -761.636f, 79.85753f)},
            {"Vent - Roof", new Vector3(334.0523f, -15.78893f, 153.2969f)},
            {"Vespucci - Helipad", new Vector3(-736.750f, -1437.750f, 5.000f)}     
        };

        public static Vector3 MAINBASEPOSITION = DroneBases.Values.ElementAt<Vector3>(0);

        // 3: Environment Methods ===================================================================================

        public static Vector3 SetRandomBasePosition(RunControl runcontrol)
        {
            return DroneBases.Values.ElementAt<Vector3>(runcontrol.random.Next(DroneBases.Count));
        }

        public static Array WEATHERS = Enum.GetValues(typeof(Weather));

        public static string SetRandomWeather(RunControl runcontrol)
        {
            Weather res_weather = (Weather)WEATHERS.GetValue(runcontrol.random.Next(WEATHERS.Length));
            World.Weather = res_weather;

            string res_string = res_weather.ToString();            
            return res_string;
        }

        public static void SetRandomTime(RunControl runcontrol)
        {
            Function.Call(Hash.SET_CLOCK_TIME,
                runcontrol.random.Next(1, 24),
                runcontrol.random.Next(1, 60),
                runcontrol.random.Next(1, 60));
        }

        public static int GetRainLevel() => (int)(Function.Call<float>(Hash.GET_RAIN_LEVEL) * 100);

        public static int GetSnowLevel() => (int)(Function.Call<float>(Hash.GET_SNOW_LEVEL) * 100);

        // 4: Save environment information to SQL ==============================================================

        public static string[] db_environment_parameters =
        {
            "EnvironmentID",
            "Weather",
            "RainLevel",
            "SnowLevel",
            "GameTime"
        };

        public static void ToSQL(TestControl testcontrol, Environment environment, SqlConnection cnn)
        {
            string sql_environment = DB.SQLCommand("Environment", db_environment_parameters);

            using (SqlCommand cmd = new SqlCommand(sql_environment, cnn))
            {
                cmd.Parameters.AddWithValue("@EnvironmentID", testcontrol.id);

                cmd.Parameters.AddWithValue("@Weather", environment.weather);
                cmd.Parameters.AddWithValue("@RainLevel", environment.rainlevel);
                cmd.Parameters.AddWithValue("@SnowLevel", environment.snowlevel);
                cmd.Parameters.AddWithValue("@GameTime", environment.gametime);

                cnn.Open();
                int res_cmd = cmd.ExecuteNonQuery();
                cnn.Close();
            }
        }
    }
}
