﻿using System.IO;
using System.Collections.Generic;
using System.Linq;
using GTA;
using GTAVisionUtils;
using BitMiracle.LibTiff.Classic;

namespace DRS
{
    public static class WriteToTiff
    {        
        // 1: Write specific buffers to a Tif file ====================================================
        public static void Colour(string path, int width, int height, byte[] color)
        {
            using (Tiff t = Tiff.Open(path, "w"))
            {
                t.SetField(TiffTag.IMAGEWIDTH, width);
                t.SetField(TiffTag.IMAGELENGTH, height);
                t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                t.SetField(TiffTag.SAMPLESPERPIXEL, 4);
                t.SetField(TiffTag.ROWSPERSTRIP, height);
                t.SetField(TiffTag.BITSPERSAMPLE, 8);
                t.SetField(TiffTag.PHOTOMETRIC, Photometric.RGB);
                t.SetField(TiffTag.COMPRESSION, Compression.JPEG);
                t.SetField(TiffTag.JPEGQUALITY, 60);
                t.SetField(TiffTag.PREDICTOR, Predictor.HORIZONTAL);
                t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT);
                t.WriteEncodedStrip(0, color, color.Length);
                t.WriteDirectory();
                t.Flush();                
            }
        }

        public static void Depth(string path, int width, int height, byte[] depth)
        {
            using(Tiff t = Tiff.Open(path, "w"))
            {
                t.SetField(TiffTag.IMAGEWIDTH, width);
                t.SetField(TiffTag.IMAGELENGTH, height);
                t.SetField(TiffTag.ROWSPERSTRIP, height);
                t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                t.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                t.SetField(TiffTag.BITSPERSAMPLE, 32);
                t.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                t.SetField(TiffTag.COMPRESSION, Compression.LZW);
                t.SetField(TiffTag.PREDICTOR, Predictor.FLOATINGPOINT);
                t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.IEEEFP);
                t.WriteEncodedStrip(0, depth, depth.Length);
                t.WriteDirectory();
                t.Flush();
            }
        }

        public static void Stencil(string path, int width, int height, byte[] stencil)
        {
            using (Tiff t = Tiff.Open(path, "w"))
            {
                t.SetField(TiffTag.IMAGEWIDTH, width);
                t.SetField(TiffTag.IMAGELENGTH, height);
                t.SetField(TiffTag.ROWSPERSTRIP, height);
                t.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                t.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                t.SetField(TiffTag.BITSPERSAMPLE, 8);
                t.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                t.SetField(TiffTag.COMPRESSION, Compression.LZW);
                t.SetField(TiffTag.PREDICTOR, Predictor.HORIZONTAL);
                t.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT);
                t.WriteEncodedStrip(0, stencil, stencil.Length);
                t.WriteDirectory();
                t.Flush();
            }
        }
        public static void BytesToTiff(RunControl runcontrol, string filename)
        {
            // Function: Write all buffers to Tif files
            // Output: Colour (1920 x 1080), depth and stencil (1280 x 720) tif images

            string basepath = Basepath(runcontrol, filename); 

            /// A: Obtain buffers

            byte[] Colour = VisionNative.GetColorBuffer();
            byte[] Depth = VisionNative.GetDepthBuffer();
            byte[] Stencil = VisionNative.GetStencilBuffer();

            Script.Wait(1);

            /// B: Formulate image paths

            string col_path = basepath + "_colour.tif";
            string dep_path = basepath + "_depth.tif";
            string ste_path = basepath + "_stencil.tif";

            /// C: Write to Tiff

            int W = Game.ScreenResolution.Width;
            int H = Game.ScreenResolution.Height;

            Script.Wait(1);

            WriteToTiff.Colour(col_path, W, H, Colour);
            WriteToTiff.Depth(dep_path, 1280, 720, Depth);
            WriteToTiff.Stencil(ste_path, 1280, 720, Stencil);
        }
        public static string Basepath(RunControl runcontrol, string filename)
        {
            // Function: Create directory (if it does exist)
            // Output: Base path

            Directory.CreateDirectory(runcontrol.outdir);                                         
            return Path.Combine(runcontrol.outdir, filename);
        }
        public static void RobustBytesToTiff(RunControl runcontrol, TestControl testcontrol, string baseimagepath)
        {
            // Function: Catches errors and regenerates BytesToTiff function 
            // Output: BytesToTiff output

            try
            {
                BytesToTiff(runcontrol, baseimagepath);
            }
            catch
            {
                PrepareGameBuffer(runcontrol, testcontrol, false);
                PrepareGameBuffer(runcontrol, testcontrol, true);
                BytesToTiff(runcontrol, baseimagepath);
            }
        }

        // 2: Prepare game buffer ====================================================================

        public static void PrepareGameBuffer(RunControl runcontrol, TestControl testcontrol, bool on)
        {
            // Function - Output: Prepares game buffers (to be written to Tif files) 

            if (on)
            {
                List<Ped> frozen_peds = World.GetNearbyPeds(runcontrol.camera.Position, 2000f).ToList();
                List<Vehicle> frozen_vehicles = World.GetNearbyVehicles(runcontrol.camera.Position, 2000f).ToList();

                frozen_vehicles.Where(x => VehicleSelection.ERRONEOUS_VEHICLE_MODELS.Contains(x.Model)).
                    ToList().ForEach(x => x.Delete());

                testcontrol.frozen_entities = new List<Entity>();
                testcontrol.frozen_entities.AddRange(frozen_peds);
                testcontrol.frozen_entities.AddRange(frozen_vehicles);
                testcontrol.frozen_entities.ForEach(x => x.FreezePosition = true);

                Script.Wait(2000);
                
                Game.TimeScale = 0;                           // [a] Slow down time TO avoid pixel drift OR return to normal  
                Game.Pause(true);                             // [b] Allow game to load sufficiently
            }
            else
            {
                Game.Pause(false);
                Game.TimeScale = 1;
                testcontrol.frozen_entities.ForEach(x => x.FreezePosition = false);                
            }                                           
        }
    }
}
