using System;

using GTA;
using GTA.Math;
using GTA.Native;

namespace DRS
{
    public class TestControl
    {
        // 1: Properties =========================================================================

        public int id;                            // Test control number

        /// A: Test

        public DateTime teststarttime;
        public DateTime testendtime;
        public int testduration;

        /// B: Camera

        public Vector3 position;
        public Vector3 rotation;
        public int fov;

        /// C: Graphics

        public Matrix world_matrix;
        public Matrix view_matrix;
        public Matrix project_matrix;

        /// D: Accident

        public double speedofimpact;
        public double angleofimpact;

        // 2: Setup =============================================================================
        
        public static TestControl Setup(RunControl runcontrol)
        {
            TestControl testcontrol = new TestControl()
            {
                id = DB.LastID(runcontrol, false) + 1,
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

    }
}

