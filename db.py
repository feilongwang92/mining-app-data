#import mysql.connector
#from mysql.connector import Error

import pymysql as mysql


def mysql_read_connection(database):
    conn = mysql.connect(host='localhost',
                                       database=database,
                                       user='root',
                                       password='Wfl920310')
    return conn


def execute(database, query):
    try:
        connection = mysql_read_connection(database)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
    except Exception as e:
        print('Error:', e)

    finally:
        cursor.close()
        connection.close()


def fetch_results(database, query, single_row=False, single_col=False):
    connection = mysql_read_connection(database)
    cursor = connection.cursor()
    cursor.execute(query)
    if single_row:
        result = cursor.fetchone()
        if single_col:
            result = result[0] if result else result
    else:
        result = list(cursor.fetchall())
        if single_col:
            result = [item[0] for item in result]
    return result


# id = 'a0be48f6b592ebcbd536f6a615be005f2066335665973d868907bd32706fe81c'
# acc = 21
# query = """
#             SELECT * from `testtest_mysql1` where id='{id}' and accuracy='{acc}'
#         """.format(id=id, acc=acc)
#
# execute(query)
# results = fetch_results(query)
# print (results)

# dbcursor = mysql.connect(host='localhost', database='cuebiq201911', user='root', password='Wfl920310').cursor()
# dbcursor.execute(query)
# results = list(dbcursor.fetchall())