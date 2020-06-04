using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data.SqlClient;

using GTA;
using GTA.Native;
using GTA.Math;
using GTAVisionUtils;

namespace DRS
{
    public class Target
    {
        public Dictionary<string, string> identifiers;    // Identifiers: "vehicleclass", "friendlyname", "primarycolour", "numberplate" 
        public Dictionary<string, int> features;          // Features: "numtyres", "numdoors", "numwindows", "capacity", "hasroof"
        public Dictionary<string, List<float>> position;  // Position and orientation: "location" (world coord), "bbox_min", "bbox_max" (model coord)
                                                          //                           "rotation (euler)", "fwd_vec", "right_vec", "up_vec" 

        public Dictionary<BB3D, List<int>> bbox;                          // 8-corner (screen coord)
        public Dictionary<string, Tuple<List<float>, List<int>>> bones;   // Vehicle bone (3D world and screen coord)

        public Damage damage;                             // Damage subclass if damaged otherwise null

        public static Target Setup(Vehicle invehicle, int damage_id)
        {
            invehicle.Model.GetDimensions(out Vector3 gmin, out Vector3 gmax);

            Target target = new Target()
            {
                identifiers = new Dictionary<string, string>()
                {
                    { "vehicleclass", invehicle.ClassType.ToString() },
                    { "friendlyname", invehicle.FriendlyName },
                    { "primarycolour", invehicle.PrimaryColor.ToString() },
                    { "numberplate", invehicle.NumberPlate }
                },

                features = new Dictionary<string, int>()
                {
                    { "numtyres", invehicle.PossibleTyres().Count },
                    { "numdoors", invehicle.PossibleDoors().Count() },
                    { "numwindows", invehicle.PossibleWindows().Count },
                    { "capacity", invehicle.PassengerSeats + 1 },
                    { "hasroof", invehicle.HasRoof.BoolToInt() }
                },

                position = new Dictionary<string, List<float>>()
                {
                    { "location", invehicle.Position.VecToList() },
                    { "bbox_min", gmin.VecToList() },
                    { "bbox_max", gmax.VecToList() },
                    { "rotation", invehicle.Rotation.VecToList()},
                    { "forward_vec", invehicle.ForwardVector.VecToList() },
                    { "right_vec", invehicle.RightVector.VecToList() },
                    { "up_vec", invehicle.UpVector.VecToList() }
                },

                bbox = Annotations.GetBB3D(invehicle),
                bones = Annotations.GetBones(invehicle, VEHICLE_BONES)
            };

            if (damage_id > 0) target.damage = Damage.Setup(damage_id, invehicle);
                                           
            return target;
        }

        public static List<string> VEHICLE_BONES = new List<string>()
        {
            "door_dside_f", "door_dside_r", "door_pside_f", "door_pside_r",
            "wheel_lf", "wheel_rf", "wheel_lr", "wheel_rr",
            "windscreen", "windscreen_r",
            "window_lf", "window_rf", "window_lr", "window_rr", 
            "bumper_f", "bumper_r",
            "bonnet", "boot",
            "headlight_l", "headlight_r", "taillight_l", "taillight_r",
        };

        
    }
}