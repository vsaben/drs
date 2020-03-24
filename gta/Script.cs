using System;
using System.Linq;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

using GTA;
using GTA.Native;
using NativeUI;

namespace DRS
{
    public class Primary : Script
    {
        private MenuPool menupool;
        private UIMenu mainmenu;
        public static RunControl runcontrol;

        public Primary()
        {
            Setup();

            Tick += Primary_Tick;
            KeyDown += Primary_KeyDown;
        }

        public void Setup()
        {
            menupool = new MenuPool();                        
            mainmenu = new UIMenu("DRS Control Panel", "");
            menupool.Add(mainmenu);

            ExperimentMenu();
            FileMenu();
        }

        public void ExperimentMenu()
        {
            UIMenu carmenu = menupool.AddSubMenu(mainmenu, "Experiment");                     

            UIMenuItem iterations = new UIMenuItem("Number of iterations");             
            carmenu.AddItem(iterations);

            carmenu.OnItemSelect += (sender, item, index) =>
            {
                if (item == iterations)
                {
                    int numberiterations = System.Convert.ToInt32(Game.GetUserInput(3));
                    runcontrol = RunControl.Setup(numberiterations);
                    Operation.Run(runcontrol);
                }
            };
        }

        public void FileMenu()
        {
            UIMenu filemenu = menupool.AddSubMenu(mainmenu, "File Directories");         

            IDictionary<string, string> fileloc = new Dictionary<string, string>
            {
                {"GTA V", "D:\\Steam\\steamapps\\common\\Grand Theft Auto V"},
                {"Image Database", "D:\\ImageDB" },
                {"DRS", "C:\\Users\\Vaughn\\projects\\work\\DRS" }
            };

            List<dynamic> filelist = fileloc.Keys.ToList<dynamic>();

            UIMenuListItem file = new UIMenuListItem("Open file locations", filelist, 0);
            filemenu.AddItem(file);

            UIMenuItem selectfile = new UIMenuItem("Open file");
            filemenu.AddItem(selectfile);

            filemenu.OnItemSelect += (sender, item, index) =>
            {
                if (item == selectfile)
                {
                    int fileindex = file.Index;
                    string filepath = fileloc.Values.ElementAt(fileindex);
                    AdditionalMethods.OpenFile(filepath);
                }
            };
        }

        // Per frame

        void Primary_Tick(object sender, EventArgs e)
        {
            if (menupool != null) menupool.ProcessMenus();
        }

        // Per key

        void Primary_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.N) AdditionalMethods.NumVehiclesDamaged();

            if (e.KeyCode == Keys.F6) AdditionalMethods.DimensionTest();

            if (e.KeyCode == Keys.F8) AdditionalMethods.NearestVehicleDamageCheck();

            if (e.KeyCode == Keys.F9) AdditionalMethods.MovePlayerToMainBase();
            
            if (e.KeyCode == Keys.F10 && !menupool.IsAnyMenuOpen())
            {
                mainmenu.Visible = !mainmenu.Visible;
            }       
        }
    }
}


