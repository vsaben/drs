using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Drawing;

using GTA;
using GTA.Native;
using GTA.Math;

using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace DRS
{
    public static class AdditionalMethods
    {
        public static void NumVehiclesDamaged()
        {
            // Function - Output: Counts the total number of vehicles (damaged) instances

            Vehicle[] allvehicles = World.GetAllVehicles();

            int numvehicles = allvehicles.Count();
            int numdamaged = allvehicles.Where(x => x.IsDamaged).Count();

            UI.Notify("Number of Vehicles [Damaged]: " + 
                numvehicles.ToString() + 
                " [" + numdamaged.ToString() + "]"); 
        }

        public static void OpenFile(string path)
        {
            // Function - Output: Open file location from GTA game menu

            Process process = new Process()
            {
                StartInfo = new ProcessStartInfo()
                {
                    FileName = "explorer.exe",
                    Arguments = path
                }
            };
            process.Start();
        }

        public static void MovePlayerToMainBase()
        {
            // WriteToTiff.PrepareGameBuffer(false);
            
            /// A: Reset player at the main base
            
            RunControl.RenderCreatedCameras(false);
            Game.Player.Character.Position = Environment.MAINBASEPOSITION;
            Game.Player.Character.IsVisible = true;

            /// B: Turn off gameplay options

            RunControl.TurnOffGameplayOptions(true);            
        }

        /*
         
        public static void Draw3DLine(Vector3 iniPos, Vector3 finPos, byte col_r = 0, byte col_g = 255, byte col_b = 0, byte col_a = 200)
        {
            // Function - Output: Draw 3D line  

            Function.Call(Hash.DRAW_LINE, new InputArgument[]
            {
                iniPos.X,
                iniPos.Y,
                iniPos.Z,
                finPos.X,
                finPos.Y,
                finPos.Z,
                (int)col_r,
                (int)col_g,
                (int)col_b,
                (int)col_a
            });
        }

        /*
        public static IDictionary<string, List<Vector3>> GetBB(Vehicle vehicle)
        {
            GTABoundingBox2 rv = new GTABoundingBox2
            {
                Min = new GTAVector2(float.PositiveInfinity, float.PositiveInfinity),
                Max = new GTAVector2(float.NegativeInfinity, float.NegativeInfinity)
            };

            List<Vector2> BB3D = new List<Vector2>() {
                { HashFunctions.Convert3dTo2d(vehicle.Position) }
            };

            vehicle.Model.GetDimensions(out Vector3 gmin, out Vector3 gmax);
            BoundingBox bbox = new SharpDX.BoundingBox((SharpDX.Vector3)new GTAVector(gmin), (SharpDX.Vector3)new GTAVector(gmax));

            foreach (SharpDX.Vector3 corner in bbox.GetCorners())
            {
                Vector3 c = new Vector3(corner.X, corner.Y, corner.Z);
                Vector3 wc = vehicle.GetOffsetInWorldCoords(c);
                Vector2 sc = HashFunctions.Convert3dTo2d(wc);

                BB3D.Add(sc);

                rv.Min.X = Math.Min(rv.Min.X, sc.X);
                rv.Min.Y = Math.Min(rv.Min.Y, sc.Y);
                rv.Max.X = Math.Max(rv.Max.X, sc.X);
                rv.Max.Y = Math.Max(rv.Max.Y, sc.Y);
            }
        
            



            // Entity matrix
            
            Vector3 rightvector = closestvehicle.RightVector;
            Vector3 forwardvector = closestvehicle.ForwardVector;
            Vector3 upvector = closestvehicle.UpVector;
            Vector3 pos = closestvehicle.Position;

            // Vehicle dimensions

            float dim_x = (float)(0.5 * dimensions.X);
            float dim_y = (float)(0.5 * dimensions.Y);
            float dim_z = (float)(0.5 * dimensions.Z);

            Vector3 dim = new Vector3(dim_x, dim_y, dim_z);

            // Front-Upper-Right (FUR) offset 

            Vector3 fur_off = dim.Y * forwardvector + dim.X * rightvector + dim.Z * upvector;

            // Corners: Front-Upper-Right (FUR), Bottom-Lower-Left (BLL)

            Vector3 fur = pos + fur_off;
            Vector3 bll = pos - fur_off;

            UI.Notify(fur.ToString());
            UI.Notify(bll.ToString());

            // Edges (8)

            Vector3 edge1 = bll;
            Vector3 edge2 = edge1 + 2 * dim_y * forwardvector;
            Vector3 edge3 = edge2 + 2 * dim_z * upvector;
            Vector3 edge4 = edge1 + 2 * dim_z * upvector;

            Vector3 edge5 = fur;
            Vector3 edge6 = edge5 - 2 * dim_y * forwardvector;
            Vector3 edge7 = edge6 - 2 * dim_z * upvector;
            Vector3 edge8 = edge5 - 2 * dim_z * upvector;

            return new List<Vector3>()

            // Draw Edges

            /// LHS
            
            Draw3DLine(edge1, edge2);
            Draw3DLine(edge1, edge4);
            Draw3DLine(edge2, edge3);
            Draw3DLine(edge3, edge4);

            /// RHS

            Draw3DLine(edge5, edge6);
            Draw3DLine(edge5, edge8);
            Draw3DLine(edge6, edge7);
            Draw3DLine(edge7, edge8);

            /// Connections

            Draw3DLine(edge1, edge7);
            Draw3DLine(edge2, edge8);
            Draw3DLine(edge3, edge5);
            Draw3DLine(edge4, edge6);
        }
        */

    }
}
