using System.IO;
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

        // 2: Functions ==========================================================================

        /// A: Write all buffers to Tif files
        public static void BytesToTiff(string filename)
        {
            string basepath = BasePath(filename); 

            //// i: Obtain buffers

            byte[] Colour = VisionNative.GetColorBuffer();
            byte[] Depth = VisionNative.GetDepthBuffer();
            byte[] Stencil = VisionNative.GetStencilBuffer();

            Script.Wait(1);

            //// ii: Formulate image paths

            string col_path = basepath + "_colour.tif";
            string dep_path = basepath + "_depth.tif";
            string ste_path = basepath + "_stencil.tif";

            //// iii: Write to Tiff

            WriteToTiff.Colour(col_path, 1280, 720, Colour);
            WriteToTiff.Depth(dep_path, 1280, 720, Depth);
            WriteToTiff.Stencil(ste_path, 1280, 720, Stencil);
        }

        public static string IMAGE_DIR() { return "D:/ImageDB/Surveillance" };
        public static string BasePath(string filename)
        {
            return Path.Combine(IMAGE_DIR(), filename);
        }

        /// B: Create a robust all buffers to Tif file function
        public static void RobustBytesToTiff(string baseimagepath)
        {
            try
            {
                BytesToTiff(baseimagepath);
            }
            catch
            {
                PrepareGameBuffer(false);
                PrepareGameBuffer(true);
                BytesToTiff(baseimagepath);
            }
        }

        /// C: Prepare game buffer

        public static void PrepareGameBuffer(bool on)
        {
            if (on)
            {
                //// i: Prepare game buffer

                Game.Pause(true);                                      // [a] Pause game
                Game.TimeScale = 0;                                    // [b] Slow down time TO avoid pixel drift
                Script.Wait(50);                                       // [c] Allow game to load sufficiently
            }
            else
            {
                //// ii: Return to normal game state

                Game.TimeScale = 1;                                    // [d] Reset timescale
                Game.Pause(false);                                     // [e] Unpause game
            }
        }
    }
}
