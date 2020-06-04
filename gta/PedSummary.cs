using System;
using System.Collections.Generic;

using GTA;
using GTA.Math;


namespace DRS
{
    public class PedSummary
    {
        public Dictionary<string, string> features;                         // Features: Gender, health status, in-vehicle   
        public Dictionary<string, List<float>> position;                    // Position and orientation: "location" (world coord), "bbox_min", "bbox_max" (model coord)
                                                                            //                           "rotation (euler)", "fwd_vec", "right_vec", "up_vec" 

        public Dictionary<BB3D, List<int>> bbox;                            // 8-corner (screen coord)
        public Dictionary<string, Tuple<List<float>, List<int>>> bones;     // Pedestrian bone (3D world and screen coord)

        public static PedSummary Setup(Ped ped)
        {
            ped.Model.GetDimensions(out Vector3 gmin, out Vector3 gmax);

            PedSummary pedsummary = new PedSummary()
            {
                features = new Dictionary<string, string>()
                    {
                        {"gender", ped.Gender.ToString()},
                        {"status", ped.IsDead ? "dead" : (ped.IsInjured ? "injured" : "healthy")},
                        {"invehicle", ped.IsInVehicle().BoolToYesNo()}
                    },

                position = new Dictionary<string, List<float>>()
                    {
                        {"location", ped.Position.VecToList()},
                        {"bbox_min", gmin.VecToList()},
                        {"bbox_max", gmax.VecToList()},
                        {"rotation", ped.Rotation.VecToList()},
                        {"forward_vec", ped.ForwardVector.VecToList()},
                        {"up_vec", ped.UpVector.VecToList()},
                        {"right_vec", ped.RightVector.VecToList()}
                    },

                bbox = Annotations.GetBB3D(ped),
                bones = Annotations.GetBones(ped, PED_BONES)
            };

            return pedsummary;
        }

        public static List<string> PED_BONES = new List<string>()
        {
            "BONETAG_PELVIS", "BONETAG_NECK","BONETAG_HEAD", "BONETAG_SPINE_ROOT", "BONETAG_SPINE",
            "BONETAG_L_THIGH", "BONETAG_L_CALF", "BONETAG_L_FOOT",
            "BONETAG_R_THIGH", "BONETAG_R_CALF", "BONETAG_R_FOOT",            
            "BONETAG_L_CLAVICLE", "BONETAG_L_UPPERARM", "BONETAG_L_FOREARM", "BONETAG_L_HAND",
            "BONETAG_R_CLAVICLE", "BONETAG_R_UPPERARM", "BONETAG_R_FOREARM", "BONETAG_R_HAND"
        };
    }
}

