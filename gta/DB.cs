using System.Data.SqlClient;
using System.Linq;
using System.Collections.Generic;

using GTA;
using GTA.Native;

namespace DRS
{
    public static class DB
    {
        // 1: Connection string =========================================================================================

        public static string ATTACHED_DB_FILENAME = "C:\\Users\\Vaughn\\projects\\work\\drs\\DRS.mdf"; 

        public static SqlConnection InitialiseCNN()
        {
            string cnn_string = "Server = (localDB)\\MSSQLLocalDB; " +
                "AttachDbFileName = " + ATTACHED_DB_FILENAME + "; " +
                "Integrated Security = True;";
            SqlConnection cnn = new SqlConnection(cnn_string);
            return cnn;
        }

        // 2: Sql command ==============================================================================================

        public static string SQLCommand(string dbname, string[] parameters)
        {
            string[] atparameters = parameters.ToList().Select(x => "@" + x).ToArray();

            string res_sqlcommand = "INSERT INTO " + dbname + "(" + string.Join(", ", parameters) + ") " +
                "VALUES(" + string.Join(", ", atparameters) + ")";

            return res_sqlcommand;
        }

        // 3: Determine Test/Type ID ===================================================================================

        public static int LastID(string control_type)
        {
            SqlConnection cnn = InitialiseCNN();                        // [a] Connection
            SqlCommand sql_cmd = SQL_CMD(cnn, control_type);            // [b] Command            
            return DBReadLastID(cnn, sql_cmd);       
        }

        public static int DBReadLastID(SqlConnection cnn, SqlCommand sql_cmd)
        {
            cnn.Open();

            SqlDataReader reader = sql_cmd.ExecuteReader();
            reader.Read();
            int res_cmd = !reader.IsDBNull(0) ? reader.GetInt32(0) : 0;

            cnn.Close();
            return res_cmd;
        }

        public static SqlCommand SQL_CMD(SqlConnection cnn, string dbname)
        {
            SqlCommand sql_cmd = new SqlCommand("Select max(" + dbname + "ID) from " + dbname, cnn);
            return sql_cmd;
        }
    }
}
