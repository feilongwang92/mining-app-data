from __future__ import print_function
import csv, time, collections, sys, os, gzip, copy, psutil
import numpy as np
from scipy import stats
from math import cos, asin, sqrt
import matplotlib.pyplot as plt
from operator import itemgetter
from random import randint, shuffle
from multiprocessing import Pool
from operator import itemgetter, add

def distance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def findlines(arg):
    namepart_i = arg[0]
    bulkname = arg[1]
    fileday = arg[2]
    print(fileday, psutil.virtual_memory().percent)
    foundlines = []
    with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\unzip\\' + fileday) as readfile:
        readCSV = csv.reader(readfile, delimiter='\t')
        for row in readCSV:
            if row[1] in bulkname:
                # foundlines.append(row)
                t = time.strftime("%m%d%H%M", time.gmtime(float(row[0]) - 18000))
                foundlines.append([row[0], bulkname[row[1]], row[2],row[3],row[4],row[5], t])
        readfile.close()

    filenamewrite = "E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part" + str(namepart_i) + fileday
    # filenamewrite = "E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part" + str(5) + fileday
    with open(filenamewrite, 'ab') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for row in foundlines:
            writeCSV.writerow(row)

        f.close()


# namelen = []
# with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part820170701.csv') as readfile:
#     readCSV = csv.reader(readfile, delimiter='\t')
#     for row in readCSV:
#         if row[1] not in namelen:
#             namelen.append(row[1])
#     readfile.close()
# print (len(set(namelen)))
# namelen = list(set(namelen))
# f = open('test.csv', 'wb')
# for name in namelen:
#     f.write(name+'\t')
# f.close()
# f = open('test.csv', 'wb')
# f.write('1'+'\t')
# f.close()


if __name__ == '__main__':
    # #############get gps#############
    # gpslist = []
    # # for fileday in os.listdir('C:\\researchdata\\CuebiqData1w'):
    # for filename in os.listdir('H:\\cuebiqseattle\\2017041700'):#('F:\\CuebiqData\\'+fileday):
    #     print (filename)
    #     with gzip.open('H:\\cuebiqseattle\\2017041700'+ '\\' + filename) as readfile:
    #     # with open('C:\\researchdata\\CuebiqData1w\\' + fileday) as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             gpslist.append((row[3][:6],row[4][:8]))
    #         readfile.close()
    #
    # # Write files
    # gpslist = list(gpslist)
    # with open('gpslist_seattle_4_14.csv', 'ab') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     writeCSV.writerow(['latitude','longitude'])
    #     for gps in gpslist:
    #         writeCSV.writerow(gps)
    #     f.close()
    # #############get gps#############

    # #############user stat acc and interval (one day)#############
    # userlist = {}
    # for filename in os.listdir('H:\\cuebiqseattle\\2017041700'):  # ('F:\\CuebiqData\\'+fileday):
    #     print(filename)
    #     with gzip.open('H:\\cuebiqseattle\\2017041700' + '\\' + filename) as readfile:
    #         # with open('C:\\researchdata\\CuebiqData1w\\' + fileday) as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             if row[1] not in userlist:
    #                 userlist[row[1]] = [row]
    #             else:
    #                 userlist[row[1]].append(row)
    #         readfile.close()
    # for name in userlist:
    #     userlist[name] = sorted(userlist[name], key=itemgetter(0))
    #
    # acclist = []
    # for name in userlist:
    #     for trace in userlist[name]:
    #         if trace[5] != '\N':
    #             acclist.append(float(trace[5]))
    # plt.hist(acclist,range(0,200,10))
    #
    # intvlist = []
    # for name in userlist:
    #     for i in range(1,len(userlist[name])):
    #         intv = int(userlist[name][i][0])-int(userlist[name][i-1][0])
    #         if intv>0 and intv < 24*3600:
    #             intvlist.append(intv)
    # plt.hist(intvlist,range(0,1000,10))
    #
    # print(len(userlist))
    #
    # plt.hist([len(userlist[name]) for name in userlist],range(0,1000,10))
    # plt.show()
    # #############user stat acc and interval#############

    ###########get name list ##############
    # filedaylist = []
    # for fileday in os.listdir('Z:\\cuebiq_psrc_2019\\original'):
    #     if int(fileday[4:6]) == 12:
    #         filedaylist.append(fileday)
    #
    # usernamelist = set()
    # monthindex = 12
    # for fileday in filedaylist:#os.listdir('Z:\\cuebiq_harvey_July_to_Dec'):
    #     print (fileday)
    #     # if int(fileday[4:6]) == monthindex:
    #     #     # Write files
    #     #     usernamelist = list(usernamelist)
    #     #     with open('Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist' + fileday[:8] + '_1.csv', 'w') as f:
    #     #         for name in usernamelist:
    #     #             f.write(name + '\n')
    #     #         f.close()
    #     #     monthindex += 1
    #     #     usernamelist = set()
    #
    #     for filename in os.listdir('Z:\\cuebiq_harvey_July_to_Dec\\' + fileday):
    #         # print(filename)
    #         with gzip.open('Z:\\cuebiq_harvey_July_to_Dec\\' + fileday + '\\' + filename) as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             name_pre = 'nonename'
    #             for row in readCSV:
    #                 name_now = row[1]
    #                 if name_now != name_pre:
    #                     # if name_pre != 'nonename':
    #                     usernamelist.add(name_pre)
    #                 name_pre = name_now
    #         readfile.close()
    #
    # # Write files
    # usernamelist = list(usernamelist)
    # with open('Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist2017' + str(monthindex) + '_1.csv', 'w') as f:
    #     for name in usernamelist:
    #         f.write(name + '\n')
    #     f.close()
    #
    #get 6 month and unique
    # filedaylist = ['usernamelist201707.csv','usernamelist201708.csv','usernamelist201709.csv',
    #                'usernamelist201710.csv','usernamelist201711.csv','usernamelist201712.csv']
    # usernamelist = set()
    # for fileday in filedaylist:#os.listdir('Z:\\cuebiq_harvey_July_to_Dec_sorted'):
    #     usernamelist_i = set()
    #     with open('Z:\\cuebiq_harvey_July_to_Dec_sorted\\' + fileday) as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             usernamelist_i.add(row[0])
    #     readfile.close()
    #     print (len(usernamelist_i))
    #     usernamelist = usernamelist.union(usernamelist_i)
    #     print (len(usernamelist))
    #
    # # Write files
    # usernamelist = list(usernamelist)
    # with open('Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist_harvey_entire.csv', 'w') as f:
    #     for name in usernamelist:
    #         f.write(name + '\n')
    #     f.close()
    #
    #shuffle name and cut into bulks
    # usernamelist = []
    # with open('Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist_harvey_entire.csv') as readfile:
    #     readCSV = csv.reader(readfile, delimiter='\t')
    #     for row in readCSV:
    #         usernamelist.append(row[0])
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # shuffle(usernamelist)
    # print ('finish shuffling!')
    #
    # wr_filename = 'Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist_harvey_entire_shuffled_labled.csv'
    # f = open(wr_filename, 'w')
    # for i in range(len(usernamelist)):
    #     f.write(usernamelist[i] + '\t' + str(i) + '\n')
    # f.close()
    #
    # # Write files
    # wr_filename = 'Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist_harvey0.csv'
    # f = open(wr_filename, 'w')
    # for i in range(len(usernamelist)):
    #     f.write(usernamelist[i] + '\t' + str(i) + '\n')
    #     if ((i+1)%5000==0):
    #         f.close()
    #         wr_filename = 'Z:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist_harvey' + str((i+1)/5000) + '.csv'
    #         f = open(wr_filename, 'w')
    #change to 1w users per file
    # usernamelist = []
    # for namepart in range(383):
    #     print ('Processing part: ', namepart)
    #     with open('J:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist\\usernamelist_harvey'
    #                       + str(namepart)+'.csv') as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             usernamelist.append(row)
    #         readfile.close()
    # # Write files
    # wr_filename = 'J:\\usernamelist_harvey0.csv'
    # f = open(wr_filename, 'w')
    # for i in range(len(usernamelist)):
    #     f.write(usernamelist[i][0] + '\t' + usernamelist[i][1] + '\n')
    #     if ((i+1)%50000==0):
    #         f.close()
    #         wr_filename = 'J:\\usernamelist_harvey' + str((i+1)/50000) + '.csv'
    #         f = open(wr_filename, 'w')
    # f.flush()
    # f.close()
    ###########get name list ##############

    # ######## sort data -- remove duplicates #######
    # ###unzip and combine
    # import shutil
    # import os
    # filedaylist = os.listdir('E:\\cuebiq_harvey_July_to_Dec')
    # for fileday in filedaylist:
    #     with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\unzip\\' + fileday[:8]+'.csv', 'wb') as wfd:
    #         for filename in os.listdir('E:\\cuebiq_harvey_July_to_Dec\\' + fileday):
    #             with gzip.open('E:\\cuebiq_harvey_July_to_Dec\\' + fileday + '\\' + filename) as fd:
    #                 shutil.copyfileobj(fd, wfd)

    # time.sleep(60*30)
    # time.sleep(60*60)

    filelist = ['part0.csv','part1.csv','part2.csv','part3.csv','part4.csv','part5-8.csv',
                'part9-12.csv','part13-16.csv','part17-20.csv','part21-24.csv','part25-28.csv',
                'part29-32.csv','part33-36.csv','part37-40.csv']
    for filename in filelist:
        print (filename)
        users10000 = []
        fw = open('E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\' + filename, 'ab')
        writeCSV = csv.writer(fw, delimiter='\t')
        with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\' + filename, 'r') as readfile:
            readCSV = csv.reader(readfile, delimiter='\t')
            name_pre = '40246'
            user = []
            for row in readCSV:
                name_now = row[1]
                if name_now != name_pre:
                    # if name_pre in resident_name:
                    if len([user]):
                        users10000.append(user)
                    if len(users10000) == 5000:
                        for traj in users10000:
                            i = 0
                            while i < len(traj) - 1:
                                if traj[i + 1][0] == traj[i][0]:
                                    if int(traj[i + 1][5]) < int(traj[i][5]):
                                    # if int(UserList[name][i + 1][4]) < int(UserList[name][i][4]):
                                        traj[i] = traj[i + 1][:]
                                    del traj[i + 1]
                                else:
                                    i += 1
                        for traj in users10000:
                            for trace in traj:
                                    writeCSV.writerow(trace)
                        users10000 = []
                    user = []
                user.append(row)
                name_pre = name_now
        fw.close()

    # ### start of procesing sort remove_duplicates #####
    # filedaylist = os.listdir('E:\\cuebiq_harvey_July_to_Dec_sorted\\unzip')
    # namepart = 68
    # while namepart <= 76:
    #     namepart += 1
    #     print ('Processing part: ', namepart)
    #     bulkname = {}
    #     with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist\\usernamelist_harvey'
    #                       + str(namepart)+'.csv') as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             bulkname[row[0]] = row[1]
    #         readfile.close()
    #     if namepart+1 <= 76:
    #         namepart += 1
    #         with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist\\usernamelist_harvey'
    #                           + str(namepart)+'.csv') as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             for row in readCSV:
    #                 bulkname[row[0]] = row[1]
    #             readfile.close()
    #     if namepart + 1 <= 76:
    #         namepart += 1
    #         with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist\\usernamelist_harvey'
    #                           + str(namepart) + '.csv') as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             for row in readCSV:
    #                 bulkname[row[0]] = row[1]
    #             readfile.close()
    #     if namepart + 1 <= 76:
    #         namepart += 1
    #         with open('E:\\cuebiq_harvey_July_to_Dec_sorted\\usernamelist\\usernamelist_harvey'
    #                           + str(namepart) + '.csv') as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             for row in readCSV:
    #                 bulkname[row[0]] = row[1]
    #             readfile.close()
    #
    #     ## bulknamelist = [bulkname for i in range(len(filedaylist))]
    #     ## zipped = zip(bulknamelist, filedaylist)
    #     ## pool = Pool(1)
    #     ## pool.map(findlines, zipped)
    #     ## pool.close()
    #     ## pool.join()
    #
    #     for fileday in filedaylist:
    #         findlines((namepart, bulkname, fileday))
    #
    #     ## filedaylist1 = filedaylist[:23]
    #     ## filedaylist1.extend(['20170922.csv','20170924.csv','20170927.csv','20170928.csv','20171001.csv',
    #     ##                      '20171004.csv','20171007.csv','20171011.csv'])
    #     ## for fileday in filedaylist1:
    #     ##     findlines((namepart, bulkname, fileday))
    #
    #     print('Now reduce')
    #     # cut into 5 * 1w
    #     bulkname = [bulkname[name] for name in bulkname.keys()]
    #     bulkname = sorted(bulkname)
    #     usernamelist = []
    #     counti = 0
    #     smallbulkname = []
    #     for name in bulkname:
    #         if (counti < 10000):
    #             smallbulkname.append(name)
    #             counti += 1
    #         else:
    #             usernamelist.append(smallbulkname)
    #             smallbulkname = []
    #             smallbulkname.append(name)
    #             counti = 1
    #     usernamelist.append(smallbulkname)  # the last one which is smaller than 100000
    #
    #
    #     while (len(usernamelist)):
    #         print (len(usernamelist))
    #         smallbulkname = usernamelist.pop()
    #         # if len(usernamelist) >= 15: continue  ########################
    #
    #         UserList = {}
    #         for name in smallbulkname: UserList[name] = []
    #
    #         for fileday in filedaylist:
    #             print(fileday, psutil.virtual_memory().percent)
    #             filename = 'E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part' + str(namepart) + fileday
    #             with open(filename) as readfile:
    #                 readCSV = csv.reader(readfile, delimiter='\t')
    #                 for row in readCSV:
    #                     if row[5] == '\N': continue
    #                     if row[1] in UserList:
    #                         UserList[row[1]].append([row[0], row[2], row[3], row[4], row[5], row[6]])
    #                 # readfile.close()
    #
    #         # sort in time order and # remove duplicates
    #         for name in UserList:
    #             UserList[name] = sorted(UserList[name], key=itemgetter(0))
    #             i = 0
    #             while i < len(UserList[name]) - 1:
    #                 if UserList[name][i + 1][0] == UserList[name][i][0]:
    #                     # if int(UserList[name][i + 1][5]) < int(UserList[name][i][5]):
    #                     if int(UserList[name][i + 1][4]) < int(UserList[name][i][4]):
    #                         UserList[name][i] = UserList[name][i + 1][:]
    #                     del UserList[name][i + 1]
    #                 else:
    #                     i += 1
    #
    #         print('end; output.', psutil.virtual_memory().percent)
    #         # write in file
    #         filenamewrite = "E:\\cuebiq_harvey_July_to_Dec_sorted\\part" + str(namepart) + ".csv"
    #         with open(filenamewrite, 'ab') as f:
    #             writeCSV = csv.writer(f, delimiter='\t')
    #             for name in UserList.keys():
    #                 for trace in UserList[name]:
    #                     # t = time.strftime("%m%d%H%M", time.gmtime(float(trace[0]) - 18000))
    #                     writeCSV.writerow([trace[0],name,trace[1],trace[2],trace[3],trace[4],trace[5]])
    #
    #         UserList = {}
    #         # remove names all ready sorted ###################
    #         if len(usernamelist) == 15 or len(usernamelist) == 10 or len(usernamelist) == 5:
    #             remaining_names = {}
    #             # for name in smallbulkname:
    #             #     remaining_names[name] = ''
    #             for bn in usernamelist:
    #                 for name in bn:
    #                     remaining_names[name] = ''
    #             for fileday in filedaylist:
    #                 print('remove lines')
    #                 remaining_lines = []
    #                 filename = 'E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part' + str(namepart) + fileday
    #                 print(filename)
    #                 with open(filename) as readfile:
    #                     readCSV = csv.reader(readfile, delimiter='\t')
    #                     for row in readCSV:
    #                         if row[1] in remaining_names:
    #                             remaining_lines.append(row)
    #                     readfile.close()
    #                 with open(filename, 'wb') as f:
    #                     f.truncate()
    #                     writeCSV = csv.writer(f, delimiter='\t')
    #                     for row in remaining_lines:
    #                         writeCSV.writerow(row)
    #                     f.close()
    #
    #     # #remove files
    #     # for fileday in filedaylist:
    #     #     filename = 'E:\\cuebiq_harvey_July_to_Dec_sorted\\parts\\part' + str(namepart) + fileday
    #     #     os.remove(filename)
    #
    #     UserList = {} #clear memory

    ### get size of file
    ## import os
    ## filelist = os.listdir('E:\\cuebiq_harvey_July_to_Dec_sorted\\unzip')
    ## for file_i in filelist:
    ##     filesize = os.path.getsize('E:\\cuebiq_harvey_July_to_Dec_sorted\\unzip\\' + file_i)
    ##     print (file_i[4:8], '\t', filesize)

    # ######## End sort data -- remove duplicates #######

    # ###############plot heat map of contradic duplicates ###################
    # diff_acc_ditance_dupl = {i: 0 for i in range(-3000, 3000, 1)}
    # with open('list_dupl_acc_diff.csv', 'r') as f:
    #     readCSV = csv.reader(f, delimiter='\t')
    #     for row in readCSV:
    #         if randint(0, 100) < 10:
    #             diff = int(float(row[1]) - float(row[0]))
    #             if diff <= -3000: diff = -3000 + 1
    #             if diff >= 3000: diff = 3000 - 1
    #             diff_acc_ditance_dupl[diff] += 1
    # sum_all = sum([diff_acc_ditance_dupl[i] for i in range(-3000, 3000, 1)])
    # diff_acc_ditance_dupl = [diff_acc_ditance_dupl[i] / float(sum_all) for i in diff_acc_ditance_dupl]
    # plt.plot([i for i in range(-3000, 3000, 1)], diff_acc_ditance_dupl)
    # plt.xlabel('Difference between low-accuracy and distance (contradict duplicated observations)')
    # plt.ylabel('Faction')
    #
    # matr_col_acc_intv = [[0 for acc in range(5000)] for intv in range(5000)]
    #
    # with open('list_dupl_acc_diff.csv', 'r') as f:
    #     readCSV = csv.reader(f, delimiter='\t')
    #     for row in readCSV:
    #         if randint(0, 100) < 2:
    #             r = int(float(row[0]))
    #             if r >= 5000: r = 4999
    #             l = int(float(row[1]))
    #             if l >= 5000: l = 4999
    #             matr_col_acc_intv[r][l] += 1

    # for i in range(5000):
    #     for j in range(5000):
    #         matr_col_acc_intv[i][j] = int(matr_col_acc_intv[i][j])#np.log2(int(matr_col_acc_intv[i][j]))#
    #
    # import seaborn as sns;
    # sns.set()
    # ax = sns.heatmap(matr_col_acc_intv)
    #
    # sum_all = 0
    # for i in range(5000):
    #     for j in range(5000):
    #         sum_all+=matr_col_acc_intv[i][j]
    #
    # rowsum = [sum(matr_col_acc_intv[i])/(float(sum_all)/100) for i in range(5000)]
    # plt.plot([i for i in range(1,5000-1)], rowsum[1:5000-1])
    # #
    # colsum = [sum([matr_col_acc_intv[i][j] for i in range(5000)])/(float(sum_all)/100) for j in range(5000)]
    # plt.plot([i for i in range(1,5000-1)], colsum[1:5000-1])
    #
    # print ('Love!')
    ###############plot heat map of contradic duplicates ###################

    # # ############distribution of accuracy##############
    # distr_acc = [0 for acc in range(1000)]
    # for filename in os.listdir('I:\\CuebiqSeattleSorted'):
    #     print(filename)
    #     with open('I:\\CuebiqSeattleSorted\\' + filename) as readfile:
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         for row in readCSV:
    #             acc = float(row[5])
    #             acc_level = 0
    #             flag = False
    #             for i in range(1000):
    #                 if acc > i * 5 and acc <= (i + 1) * 5:
    #                     acc_level = i
    #                     flag = True
    #                     break
    #             if flag == False: acc_level = 99
    #             distr_acc[acc_level] += 1
    #
    #     readfile.close()
    #
    # # Write files
    # with open('distr of accuracy seattle.csv', 'ab') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(1000):
    #         writeCSV.writerow([i * 5, distr_acc[i]])
    #     f.close()
    # #
    # # #read and plot
    # # distr_acc = [0 for acc in range(1000)]
    # # with open('distr of accuracy seattle.csv', 'r') as f:
    # #     readCSV = csv.reader(f, delimiter='\t')
    # #     i = 0
    # #     for row in readCSV:
    # #         distr_acc[i] = int(row[1])
    # #         i += 1
    # # distr_acc = [distr_acc[i] / float(sum(distr_acc)) * 100 for i in range(1000)]
    # # plt.plot([i * 5 for i in range(1, 1000)], distr_acc[1:999])
    # # ############distribution of accuracy##############

    # # ############proportion of phone observations##############
    # usernamelist = []
    # with open('usernamelist_seattle_sorted.csv') as usernamefile:
    #     readnamelist = csv.reader(usernamefile, delimiter=',')
    #     counti = 0
    #     namebulk = []
    #     for row in readnamelist:
    #         if (counti < 10000):
    #             namebulk.append(row[0])
    #             counti += 1
    #         else:
    #             usernamelist.append(namebulk)
    #             namebulk = []
    #             namebulk.append(row[0])
    #             counti = 1
    #
    #     usernamelist.append(namebulk)  # the last one which is smaller than 100000
    #     usernamefile.close()
    # print(len(usernamelist))
    # total_user = sum([len(bulk) for bulk in usernamelist])
    # print(total_user)
    #
    # distr_phone_prop = [0 for i in range(100)]
    # while (len(usernamelist)):
    #     bulkname = usernamelist.pop()
    #     print("Processing bulk: ", len(usernamelist))
    #
    #     UserList = {}
    #     for name in bulkname:
    #         UserList[name] = []
    #     print(len(UserList))
    #
    #     for filename in os.listdir('I:\\CuebiqSeattleSorted'):
    #         print(filename)
    #         with open('I:\\CuebiqSeattleSorted\\' + filename) as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             for row in readCSV:
    #                 if row[1] in UserList:
    #                     UserList[row[1]].append(row)
    #         readfile.close()
    #
    #
    #     for name in UserList:
    #         p = len([1 for row in UserList[name] if float(row[5])>100])/float(len(UserList[name]))
    #         for i in range(100):
    #             if p >= i * 0.01 and p < (i + 1) * 0.01:
    #                 distr_phone_prop[i] += 1
    #                 break
    #         if p == 1.0: distr_phone_prop[99] += 1
    #
    # distr_phone_prop = [100*distr_phone_prop[i] / float(total_user) for i in range(100)]
    #
    # #Write files
    # with open('distr of phone observ proportion.csv', 'ab') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(100):
    #         writeCSV.writerow([i, distr_phone_prop[i]])
    #     f.close()
    # # ############proportion of phone observations##############

    # # ############distribution of temporal properties (whole data set)##############
    # daylist = []
    # for fileday in os.listdir('I:\\CuebiqSeattleSorted'):
    #     daylist.append(fileday[4:8])
    # print(daylist)
    #
    # usernamelist = []
    # with open('usernamelist_seattle_sorted.csv') as usernamefile:
    #     readnamelist = csv.reader(usernamefile, delimiter=',')
    #     counti = 0
    #     namebulk = []
    #     for row in readnamelist:
    #         if (counti < 10000):
    #             namebulk.append(row[0])
    #             counti += 1
    #         else:
    #             usernamelist.append(namebulk)
    #             namebulk = []
    #             namebulk.append(row[0])
    #             counti = 1
    #     usernamelist.append(namebulk)  # the last one which is smaller than 100000
    #     usernamefile.close()
    #
    # print(len(usernamelist))
    # total_user = sum([len(bulk) for bulk in usernamelist])
    # print(total_user)
    #
    # Total_num_traj = 0
    # distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
    # num_day_obs = [0 for i in range(len(daylist) + 1)]
    # num_day_span = [0 for i in range(len(daylist) + 1)]
    # time_sparsity_within_day = [0 for i in range(48)]
    # num_time_slot = [0 for i in range(49)]
    # # store on disk
    # with open('distr_intv', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(distr_intv)):
    #         writeCSV.writerow([i, distr_intv[i]])
    #     f.close()
    # with open('num_day_obs', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(num_day_obs)):
    #         writeCSV.writerow([i, num_day_obs[i]])
    #     f.close()
    # with open('num_day_span', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(num_day_span)):
    #         writeCSV.writerow([i, num_day_span[i]])
    #     f.close()
    # with open('time_sparsity_within_day', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(time_sparsity_within_day)):
    #         writeCSV.writerow([i, time_sparsity_within_day[i]])
    #     f.close()
    # with open('num_time_slot', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(num_time_slot)):
    #         writeCSV.writerow([i, num_time_slot[i]])
    #     f.close()
    #
    # while (len(usernamelist)):
    #     bulkname = usernamelist.pop()
    #     if (len(usernamelist)>29): continue
    #     print("Processing bulk: ",len(usernamelist), time.strftime("%d-%H-%M"), psutil.virtual_memory().percent)
    #
    #     distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
    #     num_day_obs = [0 for i in range(len(daylist) + 1)]
    #     num_day_span = [0 for i in range(len(daylist) + 1)]
    #     time_sparsity_within_day = [0 for i in range(48)]
    #     num_time_slot = [0 for i in range(49)]
    #
    #     UserList = {}
    #     for name in bulkname:
    #         UserList[name] = {day: [] for day in daylist}
    #     print(len(UserList))
    #
    #     for fileday in os.listdir('I:\\CuebiqSeattleSorted'):
    #         print(fileday)
    #         with open('I:\\CuebiqSeattleSorted\\' + fileday) as readfile:
    #             readCSV = csv.reader(readfile, delimiter='\t')
    #             name_pre = 'nonename'
    #             userday = []
    #             for row in readCSV:
    #                 name_now = row[1]
    #                 if name_now != name_pre:
    #                     if name_pre in UserList:
    #                         UserList[name_pre][fileday[4:8]]=userday
    #                     userday = []
    #                 userday.append(row)
    #                 name_pre = name_now
    #         readfile.close()
    #
    #
    #     for name in UserList:
    #         dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
    #         if len(dayHavingTraj) == 0: continue
    #         num_traj = len(dayHavingTraj)
    #         Total_num_traj += num_traj
    #         num_day_obs[num_traj] += 1  # num day observed
    #         span_traj = daylist.index(dayHavingTraj[-1]) - daylist.index(dayHavingTraj[0]) + 1  # life span
    #         num_day_span[span_traj] += 1
    #         for day in dayHavingTraj:
    #             for t in range(len(UserList[name][day]) - 1):  # interval
    #                 intv = int(UserList[name][day][t + 1][0]) - int(UserList[name][day][t][0])
    #                 distr_intv[intv / 6] += 1
    #
    #     pool = Pool()
    #     results = pool.map(withinday_sparsity, [UserList[name] for name in UserList])
    #     for user in results:
    #         time_sparsity_within_day = map(add, time_sparsity_within_day, user[0])
    #         for sum_t in user[1]:
    #             num_time_slot[sum_t] += 1
    #     results=[]
    #     UserList = {}
    #     pool.close()
    #     pool.join()
    #
    #     #read from disk
    #     with open('distr_intv') as f:
    #         readCSV = csv.reader(f, delimiter='\t')
    #         i = 0
    #         for row in readCSV:
    #             distr_intv[i] += int(row[1])
    #             i += 1
    #         f.close()
    #     with open('num_day_obs') as f:
    #         readCSV = csv.reader(f, delimiter='\t')
    #         i = 0
    #         for row in readCSV:
    #             num_day_obs[i] += int(row[1])
    #             i += 1
    #         f.close()
    #     with open('num_day_span') as f:
    #         readCSV = csv.reader(f, delimiter='\t')
    #         i = 0
    #         for row in readCSV:
    #             num_day_span[i] += int(row[1])
    #             i += 1
    #         f.close()
    #     with open('time_sparsity_within_day') as f:
    #         readCSV = csv.reader(f, delimiter='\t')
    #         i = 0
    #         for row in readCSV:
    #             time_sparsity_within_day[i] += int(row[1])
    #             i += 1
    #         f.close()
    #     with open('num_time_slot') as f:
    #         readCSV = csv.reader(f, delimiter='\t')
    #         i = 0
    #         for row in readCSV:
    #             num_time_slot[i] += int(row[1])
    #             i += 1
    #         f.close()
    #     # store on disk
    #     with open('distr_intv', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for i in range(len(distr_intv)):
    #             writeCSV.writerow([i, distr_intv[i]])
    #         f.close()
    #     with open('num_day_obs', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for i in range(len(num_day_obs)):
    #             writeCSV.writerow([i, num_day_obs[i]])
    #         f.close()
    #     with open('num_day_span', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for i in range(len(num_day_span)):
    #             writeCSV.writerow([i, num_day_span[i]])
    #         f.close()
    #     with open('time_sparsity_within_day', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for i in range(len(time_sparsity_within_day)):
    #             writeCSV.writerow([i, time_sparsity_within_day[i]])
    #         f.close()
    #     with open('num_time_slot', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for i in range(len(num_time_slot)):
    #             writeCSV.writerow([i, num_time_slot[i]])
    #         f.close()
    #
    #
    #
    # distr_intv = [100*distr_intv[i] / float(sum(distr_intv)) for i in range(len(distr_intv))]
    #
    # #Write files
    # with open('temporal proporties seattle.csv', 'ab') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     writeCSV.writerow(['Total_num_traj: ', Total_num_traj])
    #     writeCSV.writerow(['total_user: ', total_user])
    #
    #     writeCSV.writerow(['number of days observed'])
    #     for i in range(len(num_day_obs)):
    #         writeCSV.writerow([i, num_day_obs[i]])
    #
    #     writeCSV.writerow(['life span'])
    #     for i in range(len(num_day_span)):
    #         writeCSV.writerow([i, num_day_span[i]])
    #
    #     writeCSV.writerow(['time sparsity within day'])
    #     for i in range(len(time_sparsity_within_day)):
    #         t_str = str(i / 2) + ':30'
    #         if i % 2 == 0: t_str = str(i / 2) + ':00'
    #         writeCSV.writerow([t_str, time_sparsity_within_day[i]])
    #
    #     writeCSV.writerow(['number of time slots'])
    #     for i in range(len(num_time_slot)):
    #         writeCSV.writerow([i, num_time_slot[i]])
    #
    #     writeCSV.writerow(['distribution of time interval between two consecutive observations (within day)'])
    #     for i in range(len(distr_intv)):
    #         writeCSV.writerow([i*6, distr_intv[i]])
    #     f.close()
    # # ############distribution of temporal propertiese (whole data set)##############

# ########### stat 1000 smaple data ###########
# ########## stat acc for 1000 data ########
# acc_gps = []
# acc_phone = []
# gps_list = []
# phone_list = []
# for name in UserList:
#     for day in UserList[name]:
#         for trace in UserList[name][day]:
#             if int(trace[5]) <= 100 and int(trace[9])>300:
#                 gps_list.append(trace)
#             elif int(trace[5]) > 100 and int(trace[9])>300:
#                 phone_list.append(trace)
#
# pre_trace = [0]*16
# for trace in gps_list:
#     if trace[6] == pre_trace[6]:
#         continue
#     else:
#         acc_gps.append(max([int(trace[5]), int(trace[8])]))
#         pre_trace = trace
#
# for trace in phone_list:
#     if trace[6] == pre_trace[6]:
#         continue
#     else:
#         acc_phone.append(max([int(trace[5]), int(trace[8])]))
#         pre_trace = trace
#
# plt.hist(acc_gps)
# plt.hist(acc_phone, range(10000))
# ########## stat acc for 1000 data ########

######stat temporal attributes##########
# def withinday_sparsity(user):
#     T_48 = [0 for i in range(48)]
#     num_slot = []
#     for day in user:
#         if len(user[day]):
#             t_48 = [0 for i in range(48)]  # within day sparsity
#             for t in range(len(user[day])):
#                 HM = time.strftime("%H%M", time.gmtime(float(user[day][t][0]) - 25200))
#                 t_48[(int(HM[:2]) * 2 + int(HM[2:]) / 30)] = 1
#             # for i in len(t_48):
#             T_48 = map(add, T_48, t_48)
#             num_slot.append(sum(t_48))
#     return (T_48, num_slot)
#
#
# daylist = [day for day in range(15, 15 + 7)]
# Total_num_traj = 0
# distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
# num_day_obs = [0 for i in range(len(daylist) + 1)]
# num_day_span = [0 for i in range(len(daylist) + 1)]
# time_sparsity_within_day = [0 for i in range(48)]
# num_time_slot = [0 for i in range(49)]
#
# usernamelist = []
# with open('F:\\CuebiqDataTestSample\\test 1000 sample PSR\\1000samplenamelist.csv') as usernamefile:
#     readnamelist = csv.reader(usernamefile, delimiter=',')
#     counti = 0
#     namebulk = []
#     for row in readnamelist:
#         if (counti < 1000):
#             namebulk.append(row[0])
#             counti += 1
#         else:
#             usernamelist.append(namebulk)
#             namebulk = []
#             namebulk.append(row[0])
#             counti = 1
#     usernamelist.append(namebulk)  # the last one which is smaller than 100000
#     usernamefile.close()
# print(len(usernamelist))
# print(sum([len(bulk) for bulk in usernamelist]))
#
# while (len(usernamelist)):
#     bulkname = usernamelist.pop()
#     print("Processing bulk: ", len(usernamelist), time.strftime("%d-%H-%M"), psutil.virtual_memory().percent)
#
#     UserList = {}
#     for name in bulkname:
#         UserList[name] = {day: [] for day in range(15, 15 + 7)}
#
#     with open('F:\\CuebiqDataTestSample\\test 1000 sample PSR\\SampleData1000PSR.csv') as readfile:
#         readCSV = csv.reader(readfile, delimiter=',')
#         next(readCSV)
#         for row in readCSV:
#             row = row[:6]  # only first 6 col useful
#             row.extend([-1, -1, -1, -1, -1, -1])
#             row[-1] = time.strftime("%d %H:%M:%S", time.gmtime(float(row[0]) - 25200))
#             name = row[1]
#             day = int((time.strftime("%d", time.gmtime(float(row[0]) - 25200))))
#             if name in UserList:
#                 UserList[name][day].append(row)
#
#         readfile.close()
#
#     for name in UserList:
#         dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
#         if len(dayHavingTraj) == 0: continue
#         num_traj = len(dayHavingTraj)
#         Total_num_traj += num_traj
#         num_day_obs[num_traj] += 1  # num day observed
#         span_traj = daylist.index(dayHavingTraj[-1]) - daylist.index(dayHavingTraj[0]) + 1  # life span
#         num_day_span[span_traj] += 1
#         for day in dayHavingTraj:
#             for t in range(len(UserList[name][day]) - 1):  # interval
#                 intv = int(UserList[name][day][t + 1][0]) - int(UserList[name][day][t][0])
#                 distr_intv[intv / 6] += 1
#
#     for name in UserList:
#         results = withinday_sparsity(UserList[name])
#         time_sparsity_within_day = map(add, time_sparsity_within_day, results[0])
#         num_time_slot[results[1]] += 1
#
# # store on disk
# with open('distr_intv', 'wb') as f:
#     writeCSV = csv.writer(f, delimiter='\t')
#     for i in range(len(distr_intv)):
#         writeCSV.writerow([i, distr_intv[i]])
#     f.close()
# with open('num_day_obs', 'wb') as f:
#     writeCSV = csv.writer(f, delimiter='\t')
#     for i in range(len(num_day_obs)):
#         writeCSV.writerow([i, num_day_obs[i]])
#     f.close()
# with open('num_day_span', 'wb') as f:
#     writeCSV = csv.writer(f, delimiter='\t')
#     for i in range(len(num_day_span)):
#         writeCSV.writerow([i, num_day_span[i]])
#     f.close()
# with open('time_sparsity_within_day', 'wb') as f:
#     writeCSV = csv.writer(f, delimiter='\t')
#     for i in range(len(time_sparsity_within_day)):
#         writeCSV.writerow([i, time_sparsity_within_day[i]])
#     f.close()
# with open('num_time_slot', 'wb') as f:
#     writeCSV = csv.writer(f, delimiter='\t')
#     for i in range(len(num_time_slot)):
#         writeCSV.writerow([i, num_time_slot[i]])
#     f.close()
#
# # ########### stat 1000 smaple data ###########

# #####stat Jinzhou's sample temporal attributes##########
#     daylist = [day for day in range(1, 1 + 61)]
#     num_day_obs = [0 for i in range(len(daylist) + 1)]
#     num_day_span = [0 for i in range(len(daylist) + 1)]
#
#     UserList = {}
#
#     with open('J:\\Harvey sample\\JZAfterProcess_69.csv') as readfile:
#         readCSV = csv.reader(readfile, delimiter=',')
#         next(readCSV)
#         for row in readCSV:
#             if row[0] not in UserList:
#                 UserList[row[0]] = [0 for i in range(len(daylist) + 1)]
#             UserList[row[0]][int(row[5]) - 212] = 1
#         readfile.close()
#
#     for name in UserList:
#         dayHavingTraj = np.sort([day for day in range(len(UserList[name])) if UserList[name][day] == 1])
#         # dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
#         if len(dayHavingTraj) == 0: continue
#         num_traj = len(dayHavingTraj)
#         num_day_obs[num_traj] += 1  # num day observed
#         span_traj = daylist.index(dayHavingTraj[-1]) - daylist.index(dayHavingTraj[0]) + 1  # life span
#         num_day_span[span_traj] += 1
#
#     # store on disk
#     with open('JZsampleTemporal.csv', 'wb') as f:
#         writeCSV = csv.writer(f, delimiter='\t')
#         writeCSV.writerow(["number day of observed#####"])
#         for i in range(len(num_day_obs)):
#             writeCSV.writerow([i, num_day_obs[i]])
#         writeCSV.writerow(["life span#####"])
#         for i in range(len(num_day_span)):
#             writeCSV.writerow([i, num_day_span[i]])
#         f.close()
#
# # ###########stat Jinzhou's sample temporal attributes###########

# # ############distribution of temporal properties (5w sample data set)##############
#     daylist = []
#     for day in range(1,29):
#         if day < 10:
#             daylist.append('0' + str(day))
#         else:
#             daylist.append(str(day))
#     Total_num_traj = 0
#     distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
#     num_day_obs = [0 for i in range(len(daylist) + 1)]
#     num_day_span = [0 for i in range(len(daylist) + 1)]
#     time_sparsity_within_day = [0 for i in range(48)]
#     num_time_slot = [0 for i in range(49)]
#
#     UserList={}
#     for fileday in daylist:
#         print(fileday)
#         with open('F:\\CuebiqDataTestSample\\test 50000 sample PSR\\50000sample201705' + fileday+'.csv') as readfile:
#             readCSV = csv.reader(readfile, delimiter='\t')
#             name_pre = 'd122e7d1b3045e44b63c35806efb29535f210b4ca670f6e3e69698127387a0d7'
#             userday = []
#             for row in readCSV:
#                 name_now = row[1]
#                 if name_now != name_pre:
#                     # if name_pre in UserList:
#                     #     UserList[name_pre][fileday[4:8]]=userday
#                     if name_pre not in UserList:
#                         UserList[name_pre]={day:[] for day in daylist}
#                     UserList[name_pre][fileday] = userday
#                     userday = []
#                 userday.append(row)
#                 name_pre = name_now
#         readfile.close()
#
#
#     for name in UserList:
#         dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
#         if len(dayHavingTraj) == 0: continue
#         num_traj = len(dayHavingTraj)
#         Total_num_traj += num_traj
#         num_day_obs[num_traj] += 1  # num day observed
#         span_traj = daylist.index(dayHavingTraj[-1]) - daylist.index(dayHavingTraj[0]) + 1  # life span
#         num_day_span[span_traj] += 1
#         for day in dayHavingTraj:
#             for t in range(len(UserList[name][day]) - 1):  # interval
#                 intv = int(UserList[name][day][t + 1][0]) - int(UserList[name][day][t][0])
#                 distr_intv[intv / 6] += 1
#         result = withinday_sparsity(UserList[name])
#         time_sparsity_within_day = map(add, time_sparsity_within_day, result[0])
#         for sum_t in result[1]:
#             num_time_slot[sum_t] += 1
#
#     # pool = Pool()
#     # results = pool.map(withinday_sparsity, [UserList[name] for name in UserList])
#     # for user in results:
#     #     time_sparsity_within_day = map(add, time_sparsity_within_day, user[0])
#     #     for sum_t in user[1]:
#     #         num_time_slot[sum_t] += 1
#
#     # distr_intv = [100*distr_intv[i] / float(sum(distr_intv)) for i in range(len(distr_intv))]
#
#     # Write files
#     with open('temporal proporties for 5w samples.csv', 'wb') as f:
#         f.truncate()
#         writeCSV = csv.writer(f, delimiter='\t')
#
#         writeCSV.writerow(['number of days observed'])
#         for i in range(len(num_day_obs)):
#             writeCSV.writerow([i, num_day_obs[i]])
#
#         writeCSV.writerow(['life span'])
#         for i in range(len(num_day_span)):
#             writeCSV.writerow([i, num_day_span[i]])
#
#         writeCSV.writerow(['time sparsity within day'])
#         for i in range(len(time_sparsity_within_day)):
#             t_str = str(i / 2) + ':30'
#             if i % 2 == 0: t_str = str(i / 2) + ':00'
#             writeCSV.writerow([t_str, time_sparsity_within_day[i]])
#
#         writeCSV.writerow(['number of time slots'])
#         for i in range(len(num_time_slot)):
#             writeCSV.writerow([i, num_time_slot[i]])
#
#         writeCSV.writerow(['distribution of time interval between two consecutive observations (within day)'])
#         for i in range(len(distr_intv)):
#             writeCSV.writerow([i * 6, distr_intv[i]])
#         f.close()
#
# # ############distribution of temporal propertiese (5w sample data set)##############