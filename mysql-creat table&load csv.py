
# currScenRecQuery = """
#                        SELECT * from 'pla_scenario_values_new2' where rid='{rId}' and qno='{qNo}'
#                     """.format(rId=1, qNo=2)
# currScenRec = db.fetch_results(currScenRecQuery, single_row=True)
#
# updateBetaAttrQuery = """
#                     UPDATE 'pla_scenario_values_new2' SET 'beta_sde' = '{beta_sde}',
#                     'beta_tts' = '{beta_tts}', 'beta_rp' = '{beta_rp}', 'beta_sdl' = '{beta_sdl}'
#                     WHERE 'pla_scenario_values_new2'.'rid' = {rId} AND 'pla_scenario_values_new2'.'qno' = '{num}';
#                     """.format(beta_sde=1, beta_tts=2, beta_rp=3, beta_sdl=4, rId=5, num=6)
# db.execute(updateBetaAttrQuery)


################### make query to fetch data ################
## start mysql service manually if failed (run->services.msc->mysql (right click) ->start)
id = 'a0be48f6b592ebcbd536f6a615be005f2066335665973d868907bd32706fe81c'
acc = 21
query = """
            SELECT * from `testtest_mysql1` where id='{id}' and accuracy='{acc}'
        """.format(id=id, acc=acc)

## two methods
## use db.py
import db
results = db.fetch_results('cuebiq201911', query)
print (results)

## use directly
import pymysql as mysql
dbconnect = mysql.connect(host='localhost', database='cuebiq201911', user='root', password='Wfl920310')
dbcursor = dbconnect.cursor()
dbcursor.execute(query)
results = dbcursor.fetchall() # is a duple of duples
[list(item) for item in results] # turn into a list of lists

dbcursor.close()
dbconnect.close()

############################ creat database ####################
import pymysql as mysql
dbconnect = mysql.connect(host='localhost', user='root', password='Wfl920310')
dbcursor = dbconnect.cursor()
dbcursor.execute("CREATE DATABASE cuebiq201801to03;") #SHOW DATABASES;  USE cuebiq201911; #
dbconnect.commit() # commit before close;  if commit, return None, else, return result
dbcursor.close()
dbconnect.close()


################### creat table and load csv ################
## creat table: two method: 1) use db.py; 2) use directly
query1 = """
            CREATE TABLE {table} (
            row_i INT NOT NULL AUTO_INCREMENT, unix_time INT NOT NULL, id VARCHAR(255) NOT NULL, type TINYINT NOT NULL, 
            latitude FLOAT NOT NULL, longitude FLOAT NOT NULL, accuracy SMALLINT NOT NULL, timezone SMALLINT NOT NULL, 
            PRIMARY KEY (row_i), INDEX id (id)) 
         """.format(table='upzip') # '{table}' return error
db.execute('cuebiq201801to03', query1)

## load csv: cannot use db.py because of the reason specified below
## Make sure to have autocommit turned on. To upload files, you need to set the local_infile parameter to 1.
usedatabase = 'cuebiq201801to03'
usetable = 'upzip'
dbconnect = mysql.connect(host='localhost', database=usedatabase, user='root', password='Wfl920310', autocommit=True, local_infile=1)
dbcursor = dbconnect.cursor()

import os
filedaylist = os.listdir('E:\\cuebiq_psrc_2019\\sorted')
filedaylist = [fileday for fileday in filedaylist if fileday.startswith('unzip20') and fileday <= 'unzip20180331.csv']

for filename in filedaylist:
    infilepath = 'E:/cuebiq_psrc_2019/sorted/' + filename #'E:/cuebiq_psrc_2019/sorted/testtest_mysql.csv'
    print (infilepath)
    query2 = """
                LOAD DATA LOCAL INFILE '{infilepath}' INTO TABLE {totable} 
                FIELDS TERMINATED BY '\\t'  
                LINES TERMINATED BY '\\n' 
                (unix_time, id, type, latitude, longitude, accuracy, timezone);
            """.format(infilepath=infilepath, totable=usetable)  # {infilepath} return error

    # db.execute('cuebiq201911', query2) # Error: InternalError(1148, u'The used command is not allowed with this MySQL version')

    dbcursor.execute(query2)
    dbconnect.commit()

dbcursor.close()
dbconnect.close()


