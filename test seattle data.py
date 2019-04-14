from __future__ import print_function
import csv, time, collections, sys, os, gzip, copy, psutil
import numpy as np
from scipy import stats
from math import cos, asin, sqrt
import matplotlib.pyplot as plt
from operator import itemgetter
from random import randint
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


def withinday_sparsity(user):
    slot_size = 60
    slot_num = 24*60/slot_size
    T_48 = [0 for _ in range(slot_num)]
    num_slot = []
    for day in user:
        if len(user[day]):
            t_48 = [0 for _ in range(slot_num)]  # within day sparsity
            for t in range(len(user[day])):
                HM = time.strftime("%H%M", time.gmtime(float(user[day][t][0]) - 25200))
                t_48[(int(HM[:2]) * 60/slot_size + int(HM[2:]) / slot_size)] = 1
            # for i in len(t_48):
            T_48 = map(add, T_48, t_48)
            num_slot.append(sum(t_48))
    return (T_48, num_slot)

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

    ############get name list ##############
    # numDay = 7
    # fileday_list = []
    # for i in range(14, numDay + 14):
    #     if i < 10:
    #         fileday_list.append('F:\\CuebiqData\\vdate=2017080' + str(i))
    #     else:
    #         fileday_list.append('F:\\CuebiqData\\vdate=201708' + str(i))
    # usernamelist = set()
    # for fileday in os.listdir('I:\\CuebiqSeattleSorted'):
    #     print(fileday)
    #     with open('I:\\CuebiqSeattleSorted\\' + fileday) as readfile:
    #         csvfile = csv.reader(readfile, delimiter='\t')
    #         for row in csvfile:
    #             usernamelist.add(row[1])
    #
    # # Write files
    # usernamelist = list(usernamelist)
    # with open('usernamelist_seattle_sorted.csv', 'w') as f:
    #     for name in usernamelist:
    #         f.write(name+'\n')
    #     f.close()
    # ###########get name list ##############

    # ######## sort data -- remove duplicates #######
    # daylist = []
    # for fileday in os.listdir('I:\\cuebiqseattle'):
    #     daylist.append(fileday[:-2])
    # print (daylist)
    #
    # usernamelist = []
    # with open('usernamelist_seattle.csv') as usernamefile:
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
    # print(len(usernamelist))
    # print(sum([len(bulk) for bulk in usernamelist]))
    #
    # num_total_observ = 0
    # num_dupl_temp = 0
    # num_dupl_contrdict = 0
    # list_dupl_acc_diff = []
    # while (len(usernamelist)):
    #     bulkname = usernamelist.pop()
    #     print("Processing bulk: ", len(usernamelist))
    #
    #     UserList = {}
    #     for name in bulkname:
    #         UserList[name] = {day: [] for day in daylist}
    #     print(len(UserList))
    #
    #     for fileday in os.listdir('I:\\cuebiqseattle'):
    #         print(fileday)
    #         for filename in os.listdir('I:\\cuebiqseattle\\'+fileday):
    #             # print(filename)
    #             with gzip.open('I:\\cuebiqseattle\\'+ fileday + '\\' + filename) as readfile:
    #                 readCSV = csv.reader(readfile, delimiter='\t')
    #                 for row in readCSV:
    #                     if row[5] == '\N': continue
    #                     name = row[1]
    #                     if name in UserList:
    #                         day = time.strftime("%Y%m%d", time.gmtime(float(row[0]) - 25200))
    #                         if day in UserList[name]:
    #                             UserList[name][day].append(row)
    #             readfile.close()
    #
    #     for name in UserList: # sort in time order
    #         for d in UserList[name]:
    #             UserList[name][d] = sorted(UserList[name][d], key=itemgetter(0))
    #     for name in UserList:  # remove duplicates
    #         for d in UserList[name]:
    #             num_total_observ += len(UserList[name][d])
    #             i = 0
    #             while i < len(UserList[name][d]) - 1:
    #                 if UserList[name][d][i + 1][0] == UserList[name][d][i][0]:
    #                     num_dupl_temp += 1
    #                     if UserList[name][d][i + 1][3] != UserList[name][d][i][3]:
    #                         dist = distance(UserList[name][d][i + 1][3], UserList[name][d][i + 1][4],UserList[name][d][i][3],UserList[name][d][i][4])
    #                         acc_max = max([float(UserList[name][d][i + 1][5]), float(UserList[name][d][i][5])])
    #                         list_dupl_acc_diff.append([dist*1000,acc_max])
    #                     if UserList[name][d][i + 1][3] == UserList[name][d][i][3] and \
    #                                     float(UserList[name][d][i + 1][5]) != float(UserList[name][d][i][5]):
    #                         num_dupl_contrdict += 1
    #                         i += 1
    #                         continue
    #                         # list_dupl_acc_diff.append([UserList[name][d][i + 1][5], UserList[name][d][i][5]])
    #                     if float(UserList[name][d][i + 1][5]) < float(UserList[name][d][i][5]):
    #                         UserList[name][d][i] = UserList[name][d][i + 1][:]
    #                     del UserList[name][d][i + 1]
    #                 else:
    #                     i += 1
    #
    #     for day in daylist:
    #         filenamewrite = "I:\\CuebiqSeattleSorted\\" + day + ".csv"
    #         with open(filenamewrite, 'ab') as f:
    #             writeCSV = csv.writer(f, delimiter='\t')
    #             for name in UserList:
    #                 for trace in UserList[name][day]:
    #                     writeCSV.writerow(trace)
    #             f.close()
    #
    #     with open('list_dupl_acc_diff.csv', 'ab') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         for pair in list_dupl_acc_diff:
    #             writeCSV.writerow(pair)
    #
    #
    # print ('num_total_observ:   ',num_total_observ)
    # print('num_dupl_temp:   ', num_dupl_temp)
    # print('num_dupl_contradic:   ', num_dupl_contrdict)
    # ######## sort data -- remove duplicates #######

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

    # ############distribution of temporal properties (whole data set)##############
    # daylist = []
    # for fileday in os.listdir('E:\\cuebiq_psrc\\cuebiq_psrc_sorted'):
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

    day_list = ['040'+str(i) if i < 10 else '04'+str(i) for i in range(4,31)]
    day_list.extend(['050' + str(i) if i < 10 else '05' + str(i) for i in range(1, 4)])
    # day_list.extend(['050'+str(i) if i < 10 else '05'+str(i) for i in range(1,32)])
    # day_list.extend(['060' + str(i)  for i in range(1, 6)])

    Total_num_traj = 0
    distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
    num_day_obs = [0 for i in range(len(day_list) + 1)]
    num_day_span = [0 for i in range(len(day_list) + 1)]
    time_sparsity_within_day = [0 for i in range(24)]
    num_time_slot = [0 for i in range(25)]
    # store on disk
    with open('distr_intv', 'wb') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for i in range(len(distr_intv)):
            writeCSV.writerow([i, distr_intv[i]])
        f.close()
    with open('num_day_obs', 'wb') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for i in range(len(num_day_obs)):
            writeCSV.writerow([i, num_day_obs[i]])
        f.close()
    with open('num_day_span', 'wb') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for i in range(len(num_day_span)):
            writeCSV.writerow([i, num_day_span[i]])
        f.close()
    with open('time_sparsity_within_day', 'wb') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for i in range(len(time_sparsity_within_day)):
            writeCSV.writerow([i, time_sparsity_within_day[i]])
        f.close()
    with open('num_time_slot', 'wb') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        for i in range(len(num_time_slot)):
            writeCSV.writerow([i, num_time_slot[i]])
        f.close()

    usernamelist = []
    with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\usernamelist_psrc_part'+str(1)+'.csv') as usernamefile:
        readnamelist = csv.reader(usernamefile, delimiter='\t')
        counti = 0
        namebulk = []
        for row in readnamelist:
            if (counti < 10000):
                namebulk.append(row[1])
                counti += 1
            else:
                usernamelist.append(namebulk)
                namebulk = []
                namebulk.append(row[1])
                counti = 1
        usernamelist.append(namebulk)  # the last one which is smaller than 100000
        usernamefile.close()
    print(len(usernamelist))
    print(sum([len(bulk) for bulk in usernamelist]))

    while (len(usernamelist)):
        bulkname = usernamelist.pop()
        print("Start processing bulk: ", len(usernamelist)+1,
              ' at time: ', time.strftime("%m%d-%H:%M"),' memory: ', psutil.virtual_memory().percent)

        distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
        num_day_obs = [0 for i in range(len(day_list) + 1)]
        num_day_span = [0 for i in range(len(day_list) + 1)]
        time_sparsity_within_day = [0 for i in range(24)]
        num_time_slot = [0 for i in range(25)]

        UserList = {}
        for name in bulkname:
            UserList[name] = {day: [] for day in day_list}

        with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\psrc_part_' + str(1) + '.csv') as readfile:
            readCSV = csv.reader(readfile, delimiter='\t')
            # next(readCSV)
            for row in readCSV:
                name = row[1]
                if name in UserList:
                    row[1] = None
                    # UserList[name][row[6][:4]].append(row)
                    if row[6][:4] in day_list: UserList[name][row[6][:4]].append(row)

    # while (len(usernamelist)):
    #     bulkname = usernamelist.pop()
    #     # if (len(usernamelist)>29): continue
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

        for name in UserList:
            dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
            if len(dayHavingTraj) == 0: continue
            num_traj = len(dayHavingTraj)
            Total_num_traj += num_traj
            num_day_obs[num_traj] += 1  # num day observed
            span_traj = day_list.index(dayHavingTraj[-1]) - day_list.index(dayHavingTraj[0]) + 1  # life span
            num_day_span[span_traj] += 1
            # for day in dayHavingTraj:
            #     for t in range(len(UserList[name][day]) - 1):  # interval
            #         intv = int(UserList[name][day][t + 1][0]) - int(UserList[name][day][t][0])
            #         distr_intv[intv / 6] += 1

        # pool = Pool()
        # results = pool.map(withinday_sparsity, [UserList[name] for name in UserList])
        # for user in results:
        #     time_sparsity_within_day = map(add, time_sparsity_within_day, user[0])
        #     for sum_t in user[1]:
        #         num_time_slot[sum_t] += 1
        # results=[]
        # UserList = {}
        # pool.close()
        # pool.join()

        # ####  read from disk
        # with open('distr_intv') as f:
        #     readCSV = csv.reader(f, delimiter='\t')
        #     i = 0
        #     for row in readCSV:
        #         distr_intv[i] += int(row[1])
        #         i += 1
        #     f.close()
        with open('num_day_obs') as f:
            readCSV = csv.reader(f, delimiter='\t')
            i = 0
            for row in readCSV:
                num_day_obs[i] += int(row[1])
                i += 1
            f.close()
        with open('num_day_span') as f:
            readCSV = csv.reader(f, delimiter='\t')
            i = 0
            for row in readCSV:
                num_day_span[i] += int(row[1])
                i += 1
            f.close()
        # with open('time_sparsity_within_day') as f:
        #     readCSV = csv.reader(f, delimiter='\t')
        #     i = 0
        #     for row in readCSV:
        #         time_sparsity_within_day[i] += int(row[1])
        #         i += 1
        #     f.close()
        # with open('num_time_slot') as f:
        #     readCSV = csv.reader(f, delimiter='\t')
        #     i = 0
        #     for row in readCSV:
        #         num_time_slot[i] += int(row[1])
        #         i += 1
        #     f.close()
        # #####   store on disk
        # with open('distr_intv', 'wb') as f:
        #     writeCSV = csv.writer(f, delimiter='\t')
        #     for i in range(len(distr_intv)):
        #         writeCSV.writerow([i, distr_intv[i]])
        #     f.close()
        with open('num_day_obs', 'wb') as f:
            writeCSV = csv.writer(f, delimiter='\t')
            for i in range(len(num_day_obs)):
                writeCSV.writerow([i, num_day_obs[i]])
            f.close()
        with open('num_day_span', 'wb') as f:
            writeCSV = csv.writer(f, delimiter='\t')
            for i in range(len(num_day_span)):
                writeCSV.writerow([i, num_day_span[i]])
            f.close()
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



    # distr_intv = [100*distr_intv[i] / float(sum(distr_intv)) for i in range(len(distr_intv))]

    #Write files
    with open('temporal proporties seattle_timeslot1hour.csv', 'ab') as f:
        writeCSV = csv.writer(f, delimiter='\t')
        # writeCSV.writerow(['Total_num_traj: ', Total_num_traj])
        # writeCSV.writerow(['total_user: ', total_user])
        #
        writeCSV.writerow(['number of days observed'])
        for i in range(len(num_day_obs)):
            writeCSV.writerow([i, num_day_obs[i]])

        writeCSV.writerow(['life span'])
        for i in range(len(num_day_span)):
            writeCSV.writerow([i, num_day_span[i]])

        # writeCSV.writerow(['time sparsity within day'])
        # for i in range(len(time_sparsity_within_day)):
        #     t_str = str(i / 2) + ':30'
        #     if i % 2 == 0: t_str = str(i / 2) + ':00'
        #     writeCSV.writerow([t_str, time_sparsity_within_day[i]])
        #
        # writeCSV.writerow(['number of time slots'])
        # for i in range(len(num_time_slot)):
        #     writeCSV.writerow([i, num_time_slot[i]])
        #
        # writeCSV.writerow(['distribution of time interval between two consecutive observations (within day)'])
        # for i in range(len(distr_intv)):
        #     writeCSV.writerow([i*6, distr_intv[i]])
        f.close()
    # ############distribution of temporal propertiese (whole data set)##############

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

# ############distribution of temporal properties (5w sample data set)##############
#     wrk_dir = "E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\downsampling\\"
#     day_list = ['040'+str(i) if i < 10 else '04'+str(i) for i in range(3,31)]
#     day_list.extend(['050'+str(i) if i < 10 else '05'+str(i) for i in range(1,32)])
#     day_list.extend(['060' + str(i)  for i in range(1, 6)])
#     # for day in range(1,29):
#     #     if day < 10:
#     #         daylist.append('0' + str(day))
#     #     else:
#     #         daylist.append(str(day))
#     Total_num_traj = 0
#     distr_intv = [0 for i in range(14400)]  # bin size: 6 seconds
#     num_day_obs = [0 for i in range(len(day_list) + 1)]
#     num_day_span = [0 for i in range(len(day_list) + 1)]
#     time_sparsity_within_day = [0 for i in range(48)]
#     num_time_slot = [0 for i in range(49)]
#
#     UserList = {}
#     namelist = [str(i) for i in range(50000)]
#     with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\downsampling\\psrc_part_0_downsampling30.csv', 'r') as readfile:
#         readCSV = csv.reader(readfile, delimiter='\t')
#         name_pre = '5988'
#         user = []
#         for row in readCSV:
#             name_now = row[1]
#             if name_now != name_pre:
#                 if name_pre in namelist:
#                     UserList[name_pre] = {day: [] for day in day_list}
#                     for trace in user:
#                         # date = time.strftime("%m%d", time.gmtime(float(trace[0]) )) #- 18000
#                         UserList[name_pre][trace[-1][:4]].append(trace)
#                 user = []
#             user.append(row)
#             name_pre = name_now
#
#     for name in UserList:
#         dayHavingTraj = np.sort([day for day in UserList[name] if len(UserList[name][day])])
#         if len(dayHavingTraj) == 0: continue
#         num_traj = len(dayHavingTraj)
#         Total_num_traj += num_traj
#         num_day_obs[num_traj] += 1  # num day observed
#         span_traj = day_list.index(dayHavingTraj[-1]) - day_list.index(dayHavingTraj[0]) + 1  # life span
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
#     with open(wrk_dir+'temporal proporties 5w downsampling30.csv', 'wb') as f:
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
#     ############End distribution of temporal properties (5w sample data set)##############

#     # ############ temporal distribution of observations within a day-compare one week ##############
#     num_observ_within_day = [[0 for _ in range(144)] for j in range(7)]
#
#     for fileday in range(7):
#         print(fileday)
#         with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted\\201704' + str(17+fileday)+'.csv') as readfile: #week of 0417
#             readCSV = csv.reader(readfile, delimiter='\t')
#             for row in readCSV:
#                 HM = time.strftime("%H%M", time.gmtime(float(row[0]) - 25200))
#                 num_observ_within_day[fileday][(int(HM[:2]) * 60/10 + int(HM[2:]) / 10)] += 1
#
#     # Write files
#     with open('temporal distribution of observations within a day.csv', 'wb') as f:
#         f.truncate()
#         writeCSV = csv.writer(f, delimiter='\t')
#
#         writeCSV.writerow(['time','Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
#         for i in range(144):
#             writeCSV.writerow([num_observ_within_day[j][i] for j in range(7)])
# # ############END temporal distribution of observations within a day-compare one week##############

    # ## ############ temporal sparsity within a day-compare one week##############
    # for fileday in range(7):
    #     print(fileday)
    #     time_sparsity_within_day = [0 for i in range(24)]
    #     UserList = {}
    #     with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted\\201704' + str(17 + fileday) + '.csv') as readfile:  # week of 0417
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         name_pre = '000'
    #         user = []
    #         for row in readCSV:
    #             name_now = row[1]
    #             if name_now != name_pre:
    #                 if name_pre not in UserList and name_pre != '000':
    #                     UserList[name_pre] = {fileday: []}
    #                     for trace in user:
    #                         UserList[name_pre][fileday].append(trace)
    #                 user = []
    #             user.append(row)
    #             name_pre = name_now
    #
    #     for name in UserList:
    #         t_48 = [0 for i in range(24)]
    #         for t in range(len(UserList[name][fileday])):
    #             HM = time.strftime("%H%M", time.gmtime(float(UserList[name][fileday][t][0]) - 25200))
    #             t_48[(int(HM[:2]) * 1 + int(HM[2:]) / 60)] = 1
    #         time_sparsity_within_day = map(add, time_sparsity_within_day, t_48)
    #
    #     with open('time sparsity within day'+str(17 + fileday)+'.csv', 'wb') as f:
    #         writeCSV = csv.writer(f, delimiter='\t')
    #         writeCSV.writerow(['total traj', len(UserList)])
    #         for i in range(len(time_sparsity_within_day)):
    #             writeCSV.writerow([i, time_sparsity_within_day[i]/float(len(UserList))])
    # ## ############ end temporal sparsity within a day-compare one week##############

    # ############ time interval at diff time of the day ###########
    # time_interval_within_day = [[] for i in range(60*24/30)]
    # UserList = {}
    # count = 10000000
    # # with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\psrc_part_0.csv') as readfile:  # week of 0417
    # with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted\\20170417.csv') as readfile:  # week of 0417
    #     readCSV = csv.reader(readfile, delimiter='\t')
    #     name_pre = '000'
    #     user = []
    #     for row in readCSV:
    #         # count -= 1
    #         # if count < 0: break
    #         name_now = row[1]
    #         if name_now != name_pre:
    #             if name_pre not in UserList and name_pre != '000':
    #                 UserList[name_pre] = {'0417':[]}
    #                 for trace in user:
    #                     UserList[name_pre]['0417'].append(trace)
    #
    #                 # UserList[name_pre] = {}
    #                 # for trace in user:
    #                 #     if trace[-1][:4] not in UserList[name_pre]: UserList[name_pre][trace[-1][:4]] = []
    #                 #     UserList[name_pre][trace[-1][:4]].append(trace)
    #
    #                 for name in UserList:
    #                     for d in UserList[name]:
    #                         for i in range(len(UserList[name][d]) - 1):
    #                             HM1 = time.strftime("%H%M", time.gmtime(float(UserList[name][d][i][0]) - 25200))
    #                             HM1 = int(HM1[:2]) * 2 + int(HM1[2:]) / 30
    #                             HM2 = time.strftime("%H%M", time.gmtime(float(UserList[name][d][i + 1][0]) - 25200))
    #                             HM2 = int(HM2[:2]) * 2 + int(HM2[2:]) / 30
    #                             if HM1 == HM2:
    #                                 intv = float(UserList[name][d][i + 1][0]) - float(UserList[name][d][i][0])
    #                                 time_interval_within_day[HM1].append(int(intv))
    #                             # intv = float(UserList[name][d][i + 1][0]) - float(UserList[name][d][i][0])
    #                             # time_interval_within_day[HM1].append(int(intv))
    #                 del UserList[name_pre]
    #
    #             user = []
    #         user.append(row)
    #         name_pre = name_now
    #
    #
    # with open('evoloving time interval within day.csv', 'wb') as f:
    #     writeCSV = csv.writer(f, delimiter='\t')
    #     for i in range(len(time_interval_within_day)):
    #         writeCSV.writerow([i, np.median(time_interval_within_day[i]) ])
    # print ("LOVE!")
    # ############ END: time interval at diff time of the day ###########

    # # ############ spatial distribution of users observed everyday ###########
    # day_list = ['040'+str(i) if i < 10 else '04'+str(i) for i in range(4,31)]
    # day_list.extend(['050'+str(i) if i < 10 else '05'+str(i) for i in range(1,32)])
    # day_list.extend(['060' + str(i)  for i in range(1, 6)])
    #
    # wf = open('user names observed everyday.csv', 'wb')
    # writeCSV = csv.writer(wf, delimiter='\t')
    # for part_num in range(10):
    #     with open('E:\\cuebiq_psrc\\cuebiq_psrc_sorted_chunked\\psrc_part_'+str(part_num)+'.csv') as readfile:
    #         print (part_num)
    #         readCSV = csv.reader(readfile, delimiter='\t')
    #         name_pre = '5988'
    #         usertraces = []
    #         for row in readCSV:
    #             name_now = row[1]
    #             if name_now != name_pre:
    #                 USER = {day: [] for day in day_list}
    #                 for trace in usertraces:
    #                     if trace[6][:4] in USER:
    #                         USER[trace[6][:4]].append(trace)
    #
    #                 dayHavingTraj = np.sort([day for day in USER if len(USER[day])])
    #                 if len(dayHavingTraj):
    #                     num_traj = len(dayHavingTraj)  # num day observed
    #                     span_traj = day_list.index(dayHavingTraj[-1]) - day_list.index(
    #                         dayHavingTraj[0]) + 1  # life span
    #                     result_spar = withinday_sparsity(USER) #within day sparsity
    #                     day_no_spar = len([1 for slotnum in result_spar[1] if slotnum == 48])
    #                     if num_traj == len(day_list) or span_traj == len(day_list) or day_no_spar:
    #                         writeCSV.writerow([name_pre, num_traj, span_traj, day_no_spar])
    #                         # print(name_pre, num_traj, span_traj)
    #
    #                 usertraces=[]
    #             usertraces.append(row)
    #             name_pre = name_now
    # wf.close()
    #
    # usernames = []
    # with open(r'C:\Users\flwang\PycharmProjects\PSRC Quebiq\user names observed everyday.csv') as rfile:
    #     readCSV = csv.reader(rfile, delimiter='\t')
    #     for row in readCSV:
    #         if row[2] == "63":
    #             usernames.append(row[0])
    #
    # wf = open(r'home of users 63 span.csv', 'wb')
    # writeCSV = csv.writer(wf, delimiter='\t')
    # with open(r'E:\cuebiq_psrc\cuebiq_psrc_sorted_chunked\cuebiq_home_psrc.csv') as rfile:
    #     rfile.next()
    #     readCSV = csv.reader(rfile, delimiter='\t')
    #     for row in readCSV:
    #         if row[0] in usernames:
    #             writeCSV.writerow(row)
    # wf.close()





