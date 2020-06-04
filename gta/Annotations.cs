using System;
using System.Collections.Generic;

using GTA;
using GTA.Math;

namespace DRS
{
    public static class Annotations
    {
        public static Dictionary<BB3D, List<int>> GetBB3D(Entity entity)
        {
            // Function - Output: Get entity 3D BB in screen co-ordinates

            Vector3 pos = entity.Position;
            Vector3 f = entity.ForwardVector;
            Vector3 r = entity.RightVector;
            Vector3 u = entity.UpVector;

            entity.Model.GetDimensions(out Vector3 gmin, out Vector3 gmax);

            Dictionary<BB3D, Vector3> bbox3D_offset = new Dictionary<BB3D, Vector3>()
            {
                {BB3D.ftl, gmin[0] * r + gmax[1] * f + gmax[2] * u },
                {BB3D.ftr, gmax[0] * r + gmax[1] * f + gmax[2] * u },
                {BB3D.fbl, gmin[0] * r + gmax[1] * f + gmin[2] * u },
                {BB3D.fbr, gmax[0] * r + gmax[1] * f + gmin[2] * u },

                {BB3D.btl, gmin[0] * r + gmin[1] * f + gmax[2] * u },
                {BB3D.btr, gmax[0] * r + gmin[1] * f + gmax[2] * u },
                {BB3D.bbl, gmin[0] * r + gmin[1] * f + gmin[2] * u },
                {BB3D.bbr, gmax[0] * r + gmin[1] * f + gmin[2] * u }
            };

            Dictionary<BB3D, List<int>> bbox = new Dictionary<BB3D, List<int>>();

            foreach (BB3D key in Enum.GetValues(typeof(BB3D)))
            {
                bbox.Add(key, new List<int>((pos + bbox3D_offset[key]).WorldToScreen()));
            }

            return bbox;
        }

        public static Dictionary<string, Tuple<List<float>, List<int>>> GetBones(Entity entity, List<string> bone_params)
        {
            // Function - Output: Get entity bones in world and screen co-ordinates

            Dictionary<string, Tuple<List<float>, List<int>>> bones = new Dictionary<string, Tuple<List<float>, List<int>>>();
            
            foreach (string bone in bone_params)
            {
                Vector3 bone_pos = entity.GetBoneCoord(bone);
                bones.Add(bone, new Tuple<List<float>, List<int>>(bone_pos.VecToList(), bone_pos.WorldToScreen()));
            }

            return bones;
        }
    }
};


