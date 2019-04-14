
import csv, time
from random import randint, shuffle

import pymysql as mysql
from multiprocessing import Pool
from multiprocessing import current_process, Lock, cpu_count



# import mysql.connector.pooling

# dbconfig = {
#     "host":"localhost",
#     "user":"root",
#     "password":"Wfl920310",
#     "database":"cuebiq201801to03",
# }
#
#
# class MySQLPool(object):
#     """
#     create a pool when connect mysql, which will decrease the time spent in
#     request connection, create connection and close connection.
#     """
#     def __init__(self, host="localhost", user="root",
#                  password="Wfl920310", database="cuebiq201801to03", pool_name="mypool",
#                  pool_size=3):
#         res = {}
#         self._host = host
#         self._user = user
#         self._password = password
#         self._database = database
#
#         res["host"] = self._host
#         res["user"] = self._user
#         res["password"] = self._password
#         res["database"] = self._database
#         self.dbconfig = res
#         self.pool = self.create_pool(pool_name=pool_name, pool_size=pool_size)
#
#     def create_pool(self, pool_name="mypool", pool_size=3):
#         """
#         Create a connection pool, after created, the request of connecting
#         MySQL could get a connection from this pool instead of request to
#         create a connection.
#         :param pool_name: the name of pool, default is "mypool"
#         :param pool_size: the size of pool, default is 3
#         :return: connection pool
#         """
#         pool = mysql.connector.pooling.MySQLConnectionPool(
#             pool_name=pool_name,
#             pool_size=pool_size,
#             pool_reset_session=True,
#             **self.dbconfig)
#         return pool
#
#     def close(self, conn, cursor):
#         """
#         A method used to close connection of mysql.
#         :param conn:
#         :param cursor:
#         :return:
#         """
#         cursor.close()
#         conn.close()
#
#     def execute(self, sql, args=None, commit=False):
#         """
#         Execute a sql, it could be with args and with out args. The usage is
#         similar with execute() function in module pymysql.
#         :param sql: sql clause
#         :param args: args need by sql clause
#         :param commit: whether to commit
#         :return: if commit, return None, else, return result
#         """
#         # get connection form connection pool instead of create one.
#         conn = self.pool.get_connection()
#         cursor = conn.cursor()
#         if args:
#             cursor.execute(sql, args)
#         else:
#             cursor.execute(sql)
#         if commit is True:
#             conn.commit()
#             self.close(conn, cursor)
#             return None
#         else:
#             res = cursor.fetchall()
#             self.close(conn, cursor)
#             return res
#
#     def executemany(self, sql, args, commit=False):
#         """
#         Execute with many args. Similar with executemany() function in pymysql.
#         args should be a sequence.
#         :param sql: sql clause
#         :param args: args
#         :param commit: commit or not.
#         :return: if commit, return None, else, return result
#         """
#         # get connection form connection pool instead of create one.
#         conn = self.pool.get_connection()
#         cursor = conn.cursor()
#         cursor.executemany(sql, args)
#         if commit is True:
#             conn.commit()
#             self.close(conn, cursor)
#             return None
#         else:
#             res = cursor.fetchall()
#             self.close(conn, cursor)
#             return res

def init(l):
    global lock
    lock = l


def parallelFuncBulk(arg):
    connect = mysql.connect(host='localhost', database='cuebiq201801to03', user='root', password='Wfl920310')
    cursor = connect.cursor()

    namepart = arg[0]
    taskbulks = arg[1]
    task = '('
    for name_i in range(len(taskbulks)-1):
        task += ('\'' + taskbulks[name_i] + '\'' + ',')
    task += ('\'' + taskbulks[-1] + '\'' + ')')
    task = "SELECT * FROM unzip where id IN {};".format(task)
    print (task)
    cursor.execute(task)
    result = cursor.fetchall()  # is a duple of duples
    result = [list(item[1:]) for item in result]  # remove first column and turn into a list of lists

    users = {name:[] for name in taskbulks}
    for row in result:
        users[row[1]].append(row)
    for name in users:
        users[name] = sorted(users[name], key = lambda x: x[0])
        i = 0
        while i < len(users[name]) - 1:
            if users[name][i + 1][0] == users[name][i][0]:
                if int(users[name][i + 1][5]) < int(users[name][i][5]):
                # if int(users[name][i + 1][4]) < int(users[name][i][4]):
                    users[name][i] = users[name][i + 1][:]
                del users[name][i + 1]
            else:
                i += 1
        i = 0
        while i < len(users[name]):
            if users[name][i][0] == -32768:# a wrong time zone; when loaded into DB, it is turned into -32768
                del users[name][i]
            else:
                i += 1
        for row in users[name]:
            row[6] = time.strftime("%y%m%d%H%M%S", time.gmtime(row[0] + row[6]))
    # write to file
    filenamewrite = "E:\\cuebiq_psrc_2019\\sorted\\part201801to03_0"+str(namepart)+".csv" if namepart < 10 else \
        "E:\\cuebiq_psrc_2019\\sorted\\part201801to03_"+str(namepart)+".csv"
    with lock:
        f = open(filenamewrite, 'ab')
        writeCSV = csv.writer(f, delimiter='\t')
        for name in users:
            for row in users[name]:
                writeCSV.writerow(row)
        f.flush()
        f.close()

    # while len(taskbulks):
    #     # query
    #     print (len(taskbulks))
    #     task = taskbulks.pop()
    #     task = "SELECT * FROM unzip where id='{}';".format(task)
    #     cursor.execute(task)
    #     result = cursor.fetchall()  # is a duple of duples
    #     result = [list(item[1:]) for item in result]  # remove first column and turn into a list of lists
    #
    #     # sort in time order and # remove duplicates
    #     result = sorted(result, key = lambda x: x[0])
    #     i = 0
    #     while i < len(result) - 1:
    #         if result[i + 1][0] == result[i][0]:
    #             if result[i + 1][5] < result[i][5]:
    #             # if result[i + 1][4] < result[i][4]:
    #                 result[i] = result[i + 1][:]
    #             del result[i + 1]
    #         else:
    #             i += 1
    #     i = 0
    #     while i < len(result):
    #         if result[i][6] == -32768:# a wrong time zone; when loaded into DB, it is turned into -32768
    #             del result[i]
    #         else:
    #             i += 1
    #
    #     for row in result:
    #         row[6] = time.strftime("%y%m%d%H%M%S", time.gmtime(row[0] + row[6]))
    #
    #     # write to file
    #     filenamewrite = "E:\\cuebiq_psrc_2019\\sorted\\part201801to03_entire.csv"
    #     with lock:
    #         f = open(filenamewrite, 'ab')
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for row in result:
    #             writeCSV.writerow(row)
    #         f.flush()
    #         f.close()
    cursor.close()
    connect.close()

if __name__ == "__main__":
    # #########get name list ##############
    # connect = mysql.connect(host='localhost', database='cuebiq201801to03', user='root', password='Wfl920310')
    # cursor = connect.cursor()
    # cursor.execute("SELECT DISTINCT id FROM unzip;")
    # result = cursor.fetchall() #a duple of duples(('a0be48f6b592ebcbd536f6a615be005f2066335665973d868907bd32706fe81c',),
    # usernamelist = [item[0] for item in result]
    # cursor.close()
    # connect.close()
    #
    # # Write files
    # shuffle(usernamelist) ## please shuffle
    # wf = open('E:\\cuebiq_psrc_2019\\sorted\\usernamelist_psrc201801to03_entire.csv', 'wb')
    # for name in usernamelist:
    #     wf.write(name + '\n')
    # wf.close()
    # numnames = len(usernamelist)
    # usernamelist = None # to save memory
    # print ("num of names", numnames)

    numnames = 304457
    ######## sort data -- remove duplicates #######
    # cut into all names into bulks; 1w per bulk
    bulksize = 5000
    for namepart in range(numnames / bulksize + 1):
        print ('Processing part: ', namepart, ' of ', numnames / bulksize + 1)
        # read 1w names from file
        bulkname = []
        cnt_i = 0
        for row in csv.reader(open('E:\\cuebiq_psrc_2019\\sorted\\usernamelist_psrc201801to03_entire.csv')):
            if cnt_i >= bulksize * namepart and cnt_i < bulksize * (namepart + 1):
                bulkname.append(row[0])
            if cnt_i >= bulksize * (namepart + 1) or cnt_i == numnames: break
            cnt_i += 1

        ## query tasks; paralleled
        # tasks = ["SELECT * FROM d20181101 where id='a0be48f6b592ebcbd536f6a615be005f2066335665973d868907bd32706fe81c';",
        #          "SELECT * FROM d20181101 where id='f6839764f3f909eefc8923b5921614785993de07f061707a0a41375fe0ebbcf0';"]
        # tasks = ["SELECT * FROM d20181101 where row_i=10;",
        #          "SELECT * FROM d20181101 where row_i=20;"]
        tasks = bulkname
        shuffle(tasks) # such that each worker get same work load
        cpu_useno = 5  # cpu_count() - 2
        parallelTaskBulks = [[] for _ in range(cpu_useno)]
        cell_i = 0
        for task in tasks:
            parallelTaskBulks[cell_i].append(task)
            cell_i += 1
            if cell_i == cpu_useno: cell_i = 0
        print("len(parallelTaskBulks), len(parallelTaskBulks[0]): ", len(parallelTaskBulks), len(parallelTaskBulks[0]))
        l = Lock()
        pool = Pool(cpu_useno, initializer=init, initargs=(l,))
        pool.map(parallelFuncBulk, [(namepart, task) for task in parallelTaskBulks])
        pool.close()
        pool.join()
