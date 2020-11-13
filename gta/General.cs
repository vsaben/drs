using System;
using System.Drawing;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;

using System.Data.Common;
using System.Configuration;

using GTA;
using GTA.Native;
using GTA.Math;
using GTAVisionUtils;

namespace DRS
{
    // 1: Explicit Enums ============================================================================

    public enum DamagePosition
    {
        // Note: NOT reflect vehicle orientation

        frontright,
        backright,
        backleft,
        frontleft         
    }
    public enum VehicleTyre
    {
        frontleft = 0,
        frontright = 1,
        middleleft = 2,
        middleright = 3,
        backleft = 4,
        backright = 5
    }


    /*
    public enum VehicleWindow
    {
        frontleft = 0, 
        frontright = 1,
        middleleft = 2, 
        middleright = 3,
        backleft = 4,
        backright = 5,
        front = 6, 
        rear = 7
    }

    public enum VehicleDoor
    {
        frontleft = 0,
        frontright = 1,
        backleft = 2, 
        backright = 3,
        hood = 4,
        trunk = 5
    }
    */

    public enum BB3D
    {
        // Function: Collect 3D bbox co-ordinates
        // Notation: (f)ront/(b)ack, (t)op/(b)ottom, (r)ight/(l)eft  

        cnt,

        ftl,
        ftr,        
        fbl,
        fbr,

        btl,
        btr,
        bbl,
        bbr
    }


    // 3: Miscellaneous Classes ======================================================================

    public static class Convert
    {
        public static int BoolToInt(this bool boolean) => boolean ? 1 : 0;
        public static string BoolToYesNo(this bool boolean) => boolean ? "Yes" : "No";
        public static string BoolToIntString(this bool boolean) => boolean ? "1" : "0";

        public static List<int> WorldToScreen(this Vector3 pos)
        {
            Vector2 standard_2D = HashFunctions.Convert3dTo2d(pos);
            Size resolution = Game.ScreenResolution;

            List<int> screen_2D = new List<int>(2){standard_2D[0] >= 0 ? (int)(standard_2D[0] * resolution.Width) : (int)standard_2D[0],
                                                   standard_2D[1] >= 0 ? (int)(standard_2D[1] * resolution.Height) : (int)standard_2D[1] };

            return screen_2D;
        }

        public static List<float> VecToList(this Vector3 vec) => new List<float>(3) { vec.X, vec.Y, vec.Z };

        public static void Populate<T>(this T[] arr, T value)
        {
            // Function - Output: Populate a list of a specified size with a default value 

            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }
        }      
    }

    public static class VehicleProp
    {
        // A: Function - Output: Determine vehicle properties [Door, Tyres, Windows]                    

        public static IOrderedEnumerable<VehicleDoor> PossibleDoors(this Vehicle vehicle)
        {
            IOrderedEnumerable<VehicleDoor> possibledoors = vehicle.GetDoors().OrderBy(x => (int)x);
            return possibledoors;
        }

        public static IOrderedEnumerable<VehicleTyre> TYRES = Enum.GetValues(typeof(VehicleTyre)).Cast<VehicleTyre>().OrderBy(x => (int)x);
        public static IList<VehicleTyre> PossibleTyres(this Vehicle vehicle)
        {
            IList<VehicleTyre> res_tyres = new List<VehicleTyre>();

            foreach (VehicleTyre tyre in TYRES)
            {
                Function.Call(Hash.SET_VEHICLE_TYRE_BURST, vehicle, (int)tyre, false, 1000f);          // [a] Burst tyre 
                bool isburst = Function.Call<bool>(Hash.IS_VEHICLE_TYRE_BURST, vehicle, (int)tyre);    // [b] Check if tyre is burst

                if (isburst)
                {
                    Function.Call(Hash.SET_VEHICLE_TYRE_FIXED, vehicle, (int)tyre);                    // [c] Fix tyre        
                    res_tyres.Add(tyre);                                                               // [d] Add tyre to possible tyre list
                }
            }

            return res_tyres;
        }

        public static IOrderedEnumerable<VehicleWindow> WINDOWS = Enum.GetValues(typeof(VehicleWindow)).Cast<VehicleWindow>().OrderBy(x => (int)x);
        public static IList<VehicleWindow> PossibleWindows(this Vehicle vehicle)
        {
            IList<VehicleWindow> res_windows = new List<VehicleWindow>();

            foreach (VehicleWindow window in WINDOWS)
            {
                bool intact = Function.Call<bool>(Hash.IS_VEHICLE_WINDOW_INTACT, vehicle, (int)window);
                if (intact) res_windows.Add(window);
            }

            return res_windows;
        }

        // B: Function - Output: Set entity grouping properties

        public static void SetPersistence(this Entity entity, bool on)
        {
            if(on) Function.Call(Hash.SET_ENTITY_AS_MISSION_ENTITY, entity, true, true);
            else entity.MarkAsNoLongerNeeded();
        }
    }
}
