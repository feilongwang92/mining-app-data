#  Env.  Anaconda2-4.0.0-Windows-x86_64; 2019.03 does not work good
#  Created on 5/20/2018 by WFL
#  gps: trace segmentation clustering
#  incremental clustering for common places
#  grid-splitting for oscillation
#  Cellular: incremental clustering for both stays and oscillations
#  Combine cellular and gps stays; do another oscillation check via incremental clustering stays
#  Excluding those having no stays and only having one stays during the entire study period
#  Improved on 04/18/2019 for efficiency
from __future__ import print_function

infile_workdir = 'E:\\cuebiq_psrc_2019\\sorted\\'
outfile_workdir = 'E:\\cuebiq_psrc_2019\\processed\\201811\\'

###  Important arguments  ###
part_num = '00'            # which data part to run
user_num_in_mem = 5000  # read how many users from the data into memory; depending on your PC memory size
dur_constr = 300        # seconds
spat_constr_gps = 0.20 # Km
spat_constr_cell = 1.0#1.0  # Km
spat_cell_split = 100   # meters; for spliting gps and cellular


import csv, time, collections, sys, os, gzip, copy, random, psutil
import numpy as np
from scipy import stats
from math import cos, asin, sqrt
import matplotlib.pyplot as plt
from operator import itemgetter
from random import randint
from multiprocessing import Pool
from operator import itemgetter, add
from itertools import combinations

from multiprocessing import current_process, Lock, cpu_count
import shutil, glob
# from geopy.distance import geodesic
def init(l):
    global lock
    lock = l

def distance(lat1, lon1, lat2, lon2):
    # return geodesic((lat1, lon1),(lat2, lon2)).km
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))
# print (distance(47.628,-122.248,47.627,-122.248))


def update_duration(user):
    for d in user.keys():
        for trace in user[d]: trace[9] = -1  # clear needed! #modify grid
        i = 0
        j = i
        while i < len(user[d]):
            if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
                dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
                for k in range(i, j, 1):
                    user[d][k][9] = dur
                break
            if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(user[d]):
                j += 1
            else:
                dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
                for k in range(i, j, 1):
                    user[d][k][9] = dur
                i = j
    return user

class cluster:
    def __init__(self):
        # self.name = name
        self.pList = []
        self.center = [0, 0]
        self.radius = 0

    def addPoint(self, point):
        self.pList.append((float(point[0]),float(point[1])))

    def updateCenter(self):
        self.center[0] = np.mean([p[0] for p in self.pList])
        self.center[1] = np.mean([p[1] for p in self.pList])

    def distance_C_point(self, point):
        self.updateCenter()
        return distance(self.center[0], self.center[1], point[0], point[1])

    def radiusC(self):
        self.updateCenter()
        r = 0
        for p in self.pList:
            d = distance(p[0], p[1], self.center[0], self.center[1])
            if d > r:
                r = d
        return r

    def has(self, point):
        if [float(point[0]), float(point[1])] in self.pList:
            return True
        return False

    def erase(self):
        self.pList = []
        self.center = [0, 0]

    def empty(self):
        if len(self.pList) == 0:
            return True
        return False


def oscillation_h1_oscill(user, dur_constr):
    user = user#arg[0]
    TimeWindow = dur_constr#arg[1]#5 * 60
    oscillgpspairlist = []

    tracelist = []
    for d in sorted(user.keys()):
        for trace in user[d]:
            dur_i = 1 if int(trace[9]) == -1 else int(trace[9])
            tracelist.append([trace[1], trace[0], dur_i, trace[6], trace[7], trace[8], 1, 1])

    # integrate: only one record representing one stay (i-i records)
    i = 0
    while i < len(tracelist) - 1:
        if tracelist[i + 1][2:5] == tracelist[i][2:5]:
            del tracelist[i + 1]
        else:
            i += 1

    flag_ppfound = False
    # get gps list from tracelist
    gpslist = [(trace[3], trace[4]) for trace in tracelist]
    # unique gps list
    uniqList = list(set(gpslist))
    # give uniq code
    tracelistno_original = [uniqList.index(gps) for gps in gpslist]
    # count duration of each gps_no
    gpsno_dur_count = {item: 0 for item in set(tracelistno_original)}
    for t in range(len(tracelist)):
        if int(tracelist[t][2]) == 0:
            gpsno_dur_count[tracelistno_original[t]] += 1
        else:
            gpsno_dur_count[tracelistno_original[t]] += int(tracelist[t][2])

    # All prepared
    oscillation_pairs = []
    t_start = 0

    # replace pong by ping; be aware that "tracelistno_original==tracelist"
    flag_find_circle = False
    while t_start < len(tracelist):
        flag_find_circle = False
        suspSequence = []
        suspSequence.append(t_start)
        for t in range(t_start + 1, len(tracelist)):  # get the suspicious sequence
            if int(tracelist[t][1]) <= int(tracelist[t_start][1]) + int(tracelist[t_start][2]) + TimeWindow:
                suspSequence.append(t)
                if tracelist[t][3:5] == tracelist[t_start][3:5]:
                    flag_find_circle = True
                    break
            else:
                break

        suspSequence_gpsno = [tracelistno_original[t] for t in suspSequence]
        # check circles
        # if len(set(suspSequence_gpsno)) < len(suspSequence_gpsno):  # implying a circle in it
        if flag_find_circle == True and len(set(suspSequence_gpsno)) != 1:  # not itself
            flag_ppfound = True
            sequence_list = [(item, gpsno_dur_count[item]) for item in set(suspSequence_gpsno)]  # ('gpsno','dur')
            sequence_list = sorted(sequence_list, key=lambda x: x[1], reverse=True)
            # get unique pairs
            oscillation_pairs = list(
                set([(sequence_list[0][0], sequence_list[i][0]) for i in range(1, len(sequence_list))]))
            t_start = suspSequence[-1]  # + 1
        else:
            t_start += 1

    # record locations of oscill pairs
    pinggps = [0, 0]
    ponggps = [0, 0]
    for pair in oscillation_pairs:
        flag_ping = 0
        flag_pong = 0
        for ii in range(len(tracelist)):  # find ping of this pair
            if tracelistno_original[ii] == pair[0]:
                pinggps = [tracelist[ii][3], tracelist[ii][4]]
                flag_ping = 1
            if tracelistno_original[ii] == pair[1]:
                ponggps = [tracelist[ii][3], tracelist[ii][4]]
                flag_pong = 1
            if flag_ping * flag_pong == 1:
                break
        pairgps = [pinggps[0], pinggps[1], ponggps[0], ponggps[1]]
        if pairgps not in oscillgpspairlist: oscillgpspairlist.append(pairgps)
    i=0 # remove -1 in oscillpair
    while i < len(oscillgpspairlist):
        if -1 in oscillgpspairlist[i]:
            del oscillgpspairlist[i]
        else:
            i += 1
    # while True:
    #
    #     for pair in oscillation_pairs:  # find all pair[1]s in list, and replace it with pair[0]
    #         for ii in range(len(tracelist)):  # find ping of this pair
    #             if tracelistno_original[ii] == pair[0]:
    #                 ping = tracelist[ii]
    #                 break
    #         for i in range(len(tracelist)):  # replace all pong with ping
    #             if tracelistno_original[i] == pair[1]:  # find pong
    #                 # numOscillRecord += 1
    #                 initime = tracelist[i][1]
    #                 dur = tracelist[i][2]  # time related attributes should remain unchanged!
    #                 tracelist[i] = ping[:]  # [:] very important!!
    #                 tracelist[i][1] = initime
    #                 tracelist[i][2] = dur
    #
    #     if flag_ppfound == False:
    #         break
    #
    #     # integrate duplicates (i-i records)
    #     i = 0
    #     while i < len(tracelist) - 1:
    #         # if tracelist[i + 1][3] == tracelist[i][3] and tracelist[i + 1][4] == tracelist[i][4] and \
    #         #     int(tracelist[i + 1][1]) - int(tracelist[i][1]) - int(tracelist[i][2]) <= TimeWindow:
    #         if tracelist[i + 1][3:5] == tracelist[i][3:5]:
    #             tracelist[i][2] = str(int(tracelist[i + 1][1]) + int(tracelist[i + 1][2]) - int(tracelist[i][1]))
    #             del tracelist[i + 1]
    #         else:
    #             i += 1

    return oscillgpspairlist


def cluster_agglomerative(user, Rc = 0.2):
    L = []
    def findClosestPair(L):
        for c in L: c.updateCenter()
        minDist = 1000000 #km
        for i in range(len(L)-1):#what if only one c
            for j in range(i+1, len(L)):
                distC2C = distance(L[i].center[0],L[i].center[1],L[j].center[0],L[j].center[1])
                # print (i,' ',j,' ',distC2C)
                if distC2C < minDist:
                    minDist = distC2C
                    CaNo = i
                    CbNo = j

        return (CaNo, CbNo)

    def mergeCaCb(CaNo, CbNo):
        Cnew = cluster()
        for p in L[CaNo].pList:
            Cnew.addPoint(p)
        for p in L[CbNo].pList:
            Cnew.addPoint(p)
        return Cnew

    def delCluser(L, CaNo, CbNo):
        L[CaNo].erase()
        L[CbNo].erase()
        while True:
            flag_del = False
            for c in L:
                if c.empty():
                    L.remove(c)
                    flag_del = True
                    break
            if flag_del == False:
                break

    for day in user.keys():
        traj = user[day]
        for trace in traj:
            if float(trace[9]) >= 5 * 60:
                p = [trace[6], trace[7]]
                if len(L) == 0:
                    Cnew = cluster()
                    Cnew.addPoint(p)
                    L.append(Cnew)
                elif not any([c.has(p) for c in L]):
                    Cnew = cluster()
                    Cnew.addPoint(p)
                    L.append(Cnew)

    while True:
        if len(L) < 2:
            break
        closestPair = findClosestPair(L)
        Cnew = mergeCaCb(closestPair[0], closestPair[1])
        if Cnew.radiusC() > Rc:
            break
        else:
            delCluser(L, closestPair[0], closestPair[1])
            L.append(Cnew)

    # cal radius
    radiusPool = [[] for _ in range(len(L))]#L[i].radius = int(1000*L[i].radiusC())
    for day in user.keys():
        for trace in user[day]:
            gps = [trace[6], trace[7]]
            for i in range(len(L)):
                if L[i].has(gps):
                    radiusPool[i].append([trace[3], trace[4]])
                    break
    for c in L:
        c.updateCenter()
    for i in range(len(radiusPool)):
        L[i].radius = 0
        for j in range(len(radiusPool[i])):
            dist_k = int(distance(L[i].center[0], L[i].center[1], radiusPool[i][j][0], radiusPool[i][j][1]) * 1000)
            if dist_k > L[i].radius: L[i].radius = dist_k

    for day in user.keys():
        for trace in user[day]:
            gps = [trace[6], trace[7]]
            for i in range(len(L)):
                if L[i].has(gps):
                    trace[6] = str(L[i].center[0])
                    trace[7] = str(L[i].center[1])
                    trace[8] = L[i].radius
                    trace[10] = 'stay_' + str(i)
                    break
    L = []
    return user


def K_meansCluster(L):
    uniqMonthGPSList = []
    for c in L:
        uniqMonthGPSList.extend(c.pList)

    Kcluster = [c.pList for c in L]
    while True:
        KcenterList = [(np.mean([p[0] for p in c]), np.mean([p[1] for p in c])) for c in Kcluster]
        Kcluster = [[] for _ in range(len(Kcluster))]

        for point in uniqMonthGPSList:
            closestCluIndex = -1
            closestDist2Clu = 1000000
            for i in range(len(KcenterList)):
                distP2C = distance(KcenterList[i][0], KcenterList[i][1], point[0], point[1])
                if closestDist2Clu > distP2C:
                    closestDist2Clu = distP2C
                    closestCluIndex = i
            Kcluster[closestCluIndex].append(point)

        i = 0
        while i < len(Kcluster):
            if len(Kcluster[i]) == 0:
                del Kcluster[i]
            else:
                i += 1

        FlagChanged = False
        for i in range(len(Kcluster)):
            cent = (np.mean([p[0] for p in Kcluster[i]]), np.mean([p[1] for p in Kcluster[i]]))
            if cent != KcenterList[i]:
                FlagChanged = True
                break

        if FlagChanged == False:
            break

    return L


def cluster_incremental(user, spat_constr, dur_constr):
    L = []

    spat_constr = spat_constr #200.0/1000 #0.2Km
    dur_constr = dur_constr#modify grid

    MonthGPSList = list(set([(trace[6], trace[7]) for d in user.keys() for trace in user[d] if int(trace[9]) >= dur_constr]))# modify grid # only cluster stays

    if len(MonthGPSList) == 0:
        return (user)

    Cnew = cluster()
    Cnew.addPoint(MonthGPSList[0])
    L.append(Cnew)
    Ccurrent = Cnew
    for i in range(1, len(MonthGPSList)):
        if Ccurrent.distance_C_point(MonthGPSList[i]) < spat_constr:
            Ccurrent.addPoint(MonthGPSList[i])
        else:
            Ccurrent = None
            for C in L:
                if C.distance_C_point(MonthGPSList[i]) < spat_constr:
                    C.addPoint(MonthGPSList[i])
                    Ccurrent = C
                    break
            if Ccurrent == None:
                Cnew = cluster()
                Cnew.addPoint(MonthGPSList[i])
                L.append(Cnew)
                Ccurrent = Cnew

    L = K_meansCluster(L)

    uniqMonthGPSList = {}
    for c in L:
        r = int(1000*c.radiusC()) #
        cent = [str(np.mean([p[0] for p in c.pList])), str(np.mean([p[1] for p in c.pList]))]
        for p in c.pList:
            uniqMonthGPSList[(str(p[0]),str(p[1]))] = (cent[0], cent[1], r)
    for d in user.keys():
        for trace in user[d]:
            if (trace[6], trace[7]) in uniqMonthGPSList:
                trace[6], trace[7], trace[8] = uniqMonthGPSList[(trace[6], trace[7])][0],\
                                               uniqMonthGPSList[(trace[6], trace[7])][1],\
                                               max(uniqMonthGPSList[(trace[6], trace[7])][2],int(trace[8]))
    return (user)


def diameterExceedCnstr(traj,i,j,spat_constr):
    #The Diameter function computes the greatest distance between any two locations in a set and compare with constraint
    # remember, distance() is costly
    loc = list(set([(round(float(traj[m][3]),5),round(float(traj[m][4]),5))  for m in range(i,j+1)]))# unique locations
    if len(loc) <= 1:
        return False
    if distance(traj[i][3],traj[i][4],traj[j][3],traj[j][4])>spat_constr: # check the first and last trace
        return True
    else:
        # guess the max distance pair; approximate distance
        pairloc = list(combinations(loc, 2))
        max_i = 0
        max_d = 0
        for i in range(len(pairloc)):
            appx_d = abs(pairloc[i][0][0] - pairloc[i][1][0]) \
                     + abs(pairloc[i][0][1] - pairloc[i][1][1])
            if appx_d > max_d:
                max_d = appx_d
                max_i = i
        if distance(pairloc[max_i][0][0], pairloc[max_i][0][1], pairloc[max_i][1][0],
                    pairloc[max_i][1][1]) > spat_constr:
            return True
        else:
            #try to reduce the size of pairloc
            max_ln_lat = (abs(pairloc[max_i][0][0] - pairloc[max_i][1][0]),
                          abs(pairloc[max_i][0][1] - pairloc[max_i][1][1]))
            m = 0
            while m < len(pairloc):
                if abs(pairloc[m][0][0] - pairloc[m][1][0]) < max_ln_lat[0] \
                        and abs(pairloc[m][0][1] - pairloc[m][1][1]) < max_ln_lat[1]:
                    del pairloc[m]
                else:
                    m += 1
            diam_list = [distance(pair[0][0], pair[0][1], pair[1][0], pair[1][1]) for pair in pairloc]
            if max(diam_list) > spat_constr:
                return True
            else:
                return False


def clusterGPS(arg):
    user = arg[0]
    dur_constr = arg[1]
    spat_constr = arg[2]

    for day in user.keys():
        traj = user[day]
        i = 0
        while (i<len(traj)-1):
            j = i
            flag = False
            while (int(traj[j][0])-int(traj[i][0])<dur_constr):#j=min k s.t. traj_k - traj_i >= dur
                j+=1
                if (j==len(traj)):
                    flag = True
                    break
            if flag:
                break
            if diameterExceedCnstr(traj,i,j,spat_constr):
                i += 1
                # print('exceed: ',i)
            else:
                # print(i)
                j_prime = j
                gps_set = set([(round(float(traj[m][3]),5),round(float(traj[m][4]),5)) for m in range(i,j+1)])
                for k in range(j_prime+1, len(traj),1): # #j: max k subject to Diameter(R,i,k)<=spat_constraint
                    if (round(float(traj[k][3]), 5), round(float(traj[k][4]), 5)) in gps_set:
                        j = k
                    elif not diameterExceedCnstr(traj,i,k, spat_constr):
                        j = k
                        gps_set.add((round(float(traj[k][3]), 5), round(float(traj[k][4]), 5)))
                    else:
                        break
                mean_lat, mean_long = str(np.mean([float(traj[k][3]) for k in range(i,j+1)])), \
                                      str(np.mean([float(traj[k][4]) for k in range(i,j+1)]))
                # traj[i][8] = 0  # give cluster radius #will give radius after agglomarative clustering
                # for k in range(i, j + 1):
                #     dist_k = int(distance(mean_lat, mean_long, traj[k][3], traj[k][4])*1000)
                #     if dist_k > traj[i][8]: traj[i][8] = dist_k
                dur = str(int(traj[j][0]) - int(traj[i][0]))  # give duration
                for k in range(i, j + 1):  # give cluster center
                    traj[k][6], traj[k][7], traj[k][9] = mean_lat, mean_long, dur
                    # traj[k][8] = traj[i][8]
                i = j+1
        user[day] = traj

    # for day in user.keys():
    #     for trace in user[day]:
    #         if float(trace[6])==-1: trace[6], trace[7] = trace[3], trace[4]


    #incremental clustering
    # user = cluster_agglomerative(user)
    user = cluster_incremental(user, spat_constr, dur_constr)

    #for those not clustered; use grid
    # modify grid
    MonthGPSList = list(set([(trace[6], trace[7], trace[8]) for d in user.keys() for trace in user[d] if int(trace[9]) >= dur_constr]))
    for day in user.keys():  # modify grid
        for trace in user[day]:
            if float(trace[6]) == -1:
                found_stay = False
                for stay_i in MonthGPSList: #first check those observations belong to a gps stay or not
                    if distance(stay_i[0], stay_i[1], trace[3], trace[4]) < spat_constr:
                        trace[6], trace[7], trace[8] = stay_i[0], stay_i[1], stay_i[2]
                        found_stay = True
                        break
                if found_stay == False:
                    trace[6] = trace[3] + '000'  # in case do not have enough digits
                    trace[7] = trace[4] + '000'
                    digits = (trace[6].split('.'))[1]
                    digits = digits[:2] + str(int(digits[2]) / 2)
                    trace[6] = (trace[6].split('.'))[0] + '.' + digits
                    # trace[6] = trace[6][:5] + str(int(trace[6][5]) / 2)  # 49.950 to 49.952 220 meters
                    digits = (trace[7].split('.'))[1]
                    digits = digits[:2] + str(int(digits[2:4]) / 25)
                    trace[7] = (trace[7].split('.'))[0] + '.' + digits
                    # trace[7] = trace[7][:7] + str(int(trace[7][7:9]) / 25)  # -122.3400 to -122.3425  180 meters

    # update duration
    user = update_duration(user)
    for d in user.keys():
        # for trace in user[d]: trace[9] = -1  # clear needed! #modify grid
        # i = 0
        # j = i
        # while i < len(user[d]):
        #     if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
        #         dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
        #         for k in range(i, j, 1):
        #             user[d][k][9] = dur
        #         break
        #     if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(user[d]):
        #         j += 1
        #     else:
        #         dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
        #         for k in range(i, j, 1):
        #             user[d][k][9] = dur
        #         i = j
        for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            if float(trace[6]) == -1: trace[9] = -1
            if float(trace[9]) == 0: trace[9] = -1

    # staysbefoscill = {(trace[6], trace[7]): None for d in user.keys() for trace in user[d] if
    #                       int(trace[9]) > dur_constr}
    # #oscillation
    # OscillationPairList = oscillation_h1_oscill(user, dur_constr) #in format: [, [pinggps[0], pinggps[1], ponggps[0], ponggps[1]]]
    # # find all pair[1]s in list, and replace it with pair[0]
    # for pair in OscillationPairList:
    #     for d in user.keys():
    #         for trace in user[d]:
    #             if trace[6] == pair[2] and trace[7] == pair[3]:
    #                 trace[6], trace[7] = pair[0], pair[1]
    # # for those new added stays, combine with gps stay
    # staysaftoscill = {(trace[6], trace[7]): None for d in user.keys() for trace in user[d] if
    #                   int(trace[9]) > dur_constr}
    # newstays = set(staysaftoscill.keys()).difference(staysbefoscill.keys())
    #

    ### commented on april 14, 2019
    # # update duration
    # for d in user.keys():
    #     user = update_duration(user)
    #     # for trace in user[d]:  # edit: not -1 at trace[6] any more.
    #     # those trace with gps as -1,-1 (not clustered) should not assign a duration
    #     #     if float(trace[6]) == -1: trace[9] = -1
    #     #     if float(trace[9]) == 0: trace[9] = -1


    for d in user.keys():
        for trace in user[d]:
            if float(trace[9]) < dur_constr: # change back keep full trajectory: do not use center for those are not stays
                trace[6], trace[7], trace[8], trace[9] = -1, -1, -1, -1  # for no stay, do not give center

    return user


def clusterPhone(arg):
    user = arg[0]
    dur_constr = arg[1]
    spat_constr = arg[2]
    # spat_constr_cell = 1.0
    L = []
    # prepare
    MonthGPSList = list(set([(trace[3], trace[4]) for d in user.keys() for trace in user[d]]))  # not Unique, just used as a container

    if len(MonthGPSList) == 0:
        return (user)

    Cnew = cluster()
    Cnew.addPoint(MonthGPSList[0])
    L.append(Cnew)
    Ccurrent = Cnew
    for i in range(1, len(MonthGPSList)):
        if Ccurrent.distance_C_point(MonthGPSList[i]) < spat_constr:
            Ccurrent.addPoint(MonthGPSList[i])
        else:
            Ccurrent = None
            for C in L:
                if C.distance_C_point(MonthGPSList[i]) < spat_constr:
                    C.addPoint(MonthGPSList[i])
                    Ccurrent = C
                    break
            if Ccurrent == None:
                Cnew = cluster()
                Cnew.addPoint(MonthGPSList[i])
                L.append(Cnew)
                Ccurrent = Cnew

    L = K_meansCluster(L)

    uniqMonthGPSList = {}
    for c in L:
        r = int(1000*c.radiusC()) #
        cent = [str(np.mean([p[0] for p in c.pList])), str(np.mean([p[1] for p in c.pList]))]
        for p in c.pList:
            uniqMonthGPSList[(str(p[0]),str(p[1]))] = (cent[0], cent[1], r)
    for d in user.keys():
        for trace in user[d]:
            if (trace[3], trace[4]) in uniqMonthGPSList:
                trace[6], trace[7], trace[8] = uniqMonthGPSList[(trace[3], trace[4])][0],\
                                               uniqMonthGPSList[(trace[3], trace[4])][1],\
                                               max(uniqMonthGPSList[(trace[3], trace[4])][2],int(trace[5]))

    # update duration
    user = update_duration(user)
    for d in user.keys():
        # for trace in user[d]: trace[9] = -1
        # i = 0
        # j = i
        # while i < len(user[d]):
        #     if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
        #         dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
        #         for k in range(i, j, 1):
        #             user[d][k][9] = dur
        #         break
        #     if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(user[d]):
        #         j += 1
        #     else:
        #         dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
        #         for k in range(i, j, 1):
        #             user[d][k][9] = dur
        #         i = j
        for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            if float(trace[6]) == -1: trace[9] = -1
            if float(trace[9]) == 0: trace[9] = -1


    #oscillation
    OscillationPairList = oscillation_h1_oscill(user, dur_constr) #in format: [, [pinggps[0], pinggps[1], ponggps[0], ponggps[1]]]
    # find all pair[1]s in list, and replace it with pair[0]
    for pair in OscillationPairList:
        for d in user.keys():
            for trace in user[d]:
                if trace[6] == pair[2] and trace[7] == pair[3]:
                    trace[6], trace[7] = pair[0], pair[1]

    # update duration
    user = update_duration(user)
    for d in user.keys():
        for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            if float(trace[6]) == -1: trace[9] = -1
            if float(trace[9]) == 0: trace[9] = -1

    # # ijij oscillation
    # OscillationPairList = oscillation_ijij_revision(user, dur_constr)
    # # in format: [, [pinggps[0], pinggps[1], ponggps[0], ponggps[1]]]
    # # find all pair[1]s in list, and replace it with pair[0]
    # for pair in OscillationPairList:
    #     for d in user.keys():
    #         for trace in user[d]:
    #             if trace[6] == pair[2] and trace[7] == pair[3]:
    #                 trace[6], trace[7] = pair[0], pair[1]
    #
    # # update duration
    # for d in user.keys():
    #     user = update_duration(user)

    return user


def combineGPSandPhoneStops(arg):
    user_gps = arg[0]
    user_cell = arg[1]
    dur_constr = arg[2]
    spat_constr_gps = arg[3]
    spat_cell_split = arg[4]

    # combine cellular stay if it is close to a gps stay
    cell_stays = list(set([(trace[6],trace[7]) for d in user_cell for trace in user_cell[d] if int(trace[9]) >= dur_constr]))
    gps_stays = list(set([(trace[6],trace[7]) for d in user_gps for trace in user_gps[d] if int(trace[9]) >= dur_constr]))
    pairs_close = set()
    for cell_stay in cell_stays:
        for gps_stay in gps_stays:
            if distance(cell_stay[0],cell_stay[1],gps_stay[0],gps_stay[1])<=spat_constr_gps:
                pairs_close.add((gps_stay[0],gps_stay[1],cell_stay[0],cell_stay[1]))
                break
    # find all pair[1]s in list, and replace it with pair[0]
    for pair in list(pairs_close):
        for d in user_cell.keys():
            for trace in user_cell[d]:
                if trace[6] == pair[2] and trace[7] == pair[3]:
                    trace[5], trace[6], trace[7] = 99, pair[0], pair[1] #pretend as gps

    user = user_gps
    for d in user.keys():
        if len(user_cell[d]):
            user[d].extend(user_cell[d])
            user[d] = sorted(user[d], key=itemgetter(0))

    # address oscillation
    OscillationPairList = oscillation_h1_oscill(user, dur_constr)  # in format: [, [ping[0], ping[1], pong[0], pong[1]]]
    gpslist_temp = {(trace[6], trace[7]):int(trace[5]) for d in user.keys() for trace in user[d]}
    for pair_i in range(len(OscillationPairList)):#when replaced, can only replaced with a gps stay
        if gpslist_temp[(OscillationPairList[pair_i][0],OscillationPairList[pair_i][1])] <= spat_constr_gps:
            OscillationPairList[pair_i] = [OscillationPairList[pair_i][2],OscillationPairList[pair_i][3],
                                           OscillationPairList[pair_i][0],OscillationPairList[pair_i][1]]
    # find all pair[1]s in list, and replace it with pair[0]
    for pair in OscillationPairList:
        for d in user.keys():
            for trace in user[d]:
                if (trace[6], trace[7]) == (trace[2], trace[3]):
                    trace[6], trace[7] = pair[0], pair[1]
    # update duration
    user = update_duration(user)
    for d in user:
        for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            if float(trace[6]) == -1: trace[9] = -1

    for d in user:
        phone_index = [k for k in range(len(user[d])) if int(user[d][k][5]) > spat_cell_split]
        if len(phone_index) == 0:  # if no phone trace
            continue
        for i in range(len(user[d])):
            if int(user[d][i][5]) > spat_cell_split and int(user[d][i][9]) < dur_constr:  # passing phone observ
                user[d][i].append('checked')
        # combine consecutive obsv on a phone stay into two observ
        i = min(phone_index)  # i has to be a phone index
        j = i + 1
        while i < len(user[d]) - 1:
            if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
                for k in range(i + 1, j - 1, 1):
                    user[d][k] = []
                break
            if int(user[d][j][5]) > spat_cell_split and user[d][j][6] == user[d][i][6] \
                    and user[d][j][7] == user[d][i][7] and j < len(user[d]):
                j += 1
            else:
                for k in range(i + 1, j - 1, 1):
                    user[d][k] = []
                phone_index = [k for k in range(j, len(user[d])) if int(user[d][k][5]) > spat_cell_split]
                if len(phone_index) < 3:  # if no phone trace
                    break
                i = min(phone_index)  ##i has to be a phone index
                j = i + 1
        i = 0  # remove []
        while i < len(user[d]):
            if len(user[d][i]) == 0:
                del user[d][i]
            else:
                i += 1
        # adress phone stay one by one
        flag_changed = True
        phone_list_check = []
        while (flag_changed):
            # print('while........')
            flag_changed = False
            gps_list = []
            phone_list = []
            for i in range(len(user[d])):
                if int(user[d][i][5]) <= spat_cell_split:#or user[d][i][2] == 'addedphonestay': #changed on 0428
                    gps_list.append(user[d][i])
                else:
                    phone_list.append(user[d][i])

            # # update gps stay
            # i = 0
            # j = i
            # while i < len(gps_list):
            #     if j >= len(gps_list):  # a day ending with a stay, j goes beyond the last observation
            #         dur = str(int(gps_list[j - 1][0]) - int(gps_list[i][0]))
            #         for k in range(i, j, 1):
            #             gps_list[k][9] = dur
            #         break
            #     if gps_list[j][6] == gps_list[i][6] and gps_list[j][7] == gps_list[i][7] and j < len(
            #             gps_list):
            #         j += 1
            #     else:
            #         dur = str(int(gps_list[j - 1][0]) - int(gps_list[i][0]))
            #         for k in range(i, j, 1):
            #             gps_list[k][9] = dur
            #         i = j
            # for trace in gps_list:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            #     if int(trace[6]) == -1: trace[9] = -1
            # if len(gps_list) == 1: gps_list[0][9] = -1

            phone_list.extend(phone_list_check)
            # when updating duration for phone stay, we have to put back passing obs
            phone_list = sorted(phone_list, key=itemgetter(0))
            # update phone stay
            i = 0
            j = i
            while i < len(phone_list):
                if j >= len(phone_list):  # a day ending with a stay, j goes beyond the last observation
                    dur = str(int(phone_list[j - 1][0]) - int(phone_list[i][0]))
                    for k in range(i, j, 1):
                        if int(phone_list[k][9]) >= dur_constr:
                        # we don't want to change a pssing into a stay; as  we have not process the combine this stay
                        # this is possible when a stay that prevents two passing is mergeed into gps as gps points
                            phone_list[k][9] = dur
                    break
                if phone_list[j][6] == phone_list[i][6] and phone_list[j][7] == phone_list[i][7] and j < len(
                        phone_list):
                    j += 1
                else:
                    dur = str(int(phone_list[j - 1][0]) - int(phone_list[i][0]))
                    for k in range(i, j, 1):
                        if int(phone_list[k][9]) >= dur_constr:
                            phone_list[k][9] = dur
                    i = j
            for trace in phone_list:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
                if float(trace[6]) == -1: trace[9] = -1
            if len(phone_list) == 1: phone_list[0][9] = -1

            # update check lable
            for i in range(len(phone_list)):
                if int(phone_list[i][5]) > spat_cell_split and int(phone_list[i][9]) < dur_constr \
                        and phone_list[i][-1] != 'checked':
                    # passing phone observ
                    phone_list[i].append('checked')

            # put those not checked together with gps
            user[d] = gps_list
            phone_list_check = []
            for i in range(len(phone_list)):
                if phone_list[i][-1] == 'checked':
                    phone_list_check.append(phone_list[i])
                else:
                    user[d].append(phone_list[i])
            user[d] = sorted(user[d], key=itemgetter(0))

            # find a stay which is not checked
            flag_phonestay_notchecked = False
            phonestay_left, phonestay_right = -1, -1
            for i in range(max(0, phonestay_right+1), len(user[d])):
                phonestay_left, phonestay_right = -1, -1
                if int(user[d][i][5]) > spat_cell_split \
                        and int(user[d][i][9]) >= dur_constr and user[d][i][-1] != 'checked':
                    phonestay_left = phonestay_right
                    phonestay_right = i
                if phonestay_left != -1 and phonestay_right != -1 \
                        and user[d][phonestay_left][9] == user[d][phonestay_right][9]:
                    flag_phonestay_notchecked = True

                ## modified on 04152019
                if flag_phonestay_notchecked == False or len(phone_list) == 0: # if all phone observation are checked, end
                    break
                # if they are not two consecutive observation
                if phonestay_right != phonestay_left + 1:  # attention: only phonestay_left is addressed
                    # not consecutive two observations
                    if any([int(user[d][j][9]) >= dur_constr for j in range(phonestay_left + 1, phonestay_right, 1)]):
                        # found a gps stay in betw
                        # print('23: found a gps stay in betw, just use one gps stay trade one phone stay')
                        temp = user[d][phonestay_left][6:]
                        user[d][phonestay_left][6:] = [-1, -1, -1, -1, -1, -1]  # phone disappear
                        # user[d][phonestay_left].extend(temp)
                        user[d][phonestay_left].append('checked')
                        # del user[d][phonestay_left]  # phone disappear
                        flag_changed = True
                    else:  # find close gps
                        # print('24: do not found a gps stay in betw')
                        phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                       int(user[d][phonestay_right][5])])
                        if all([(phone_uncernt + int(user[d][j][5])) > 1000 * distance(user[d][j][3], user[d][j][4],
                                                                                       user[d][phonestay_left][6],
                                                                                       user[d][phonestay_left][7])
                                for j in range(phonestay_left + 1, phonestay_right, 1)]):
                            #total uncerty larger than distance
                            # this case should be rare, as those close gps may be clustered
                            # print('241: all gps falling betw are close with phone stay')
                            temp = user[d][phonestay_left][3:]  # copy neighbor gps
                            user[d][phonestay_left][3:] = user[d][phonestay_left + 1][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            flag_changed = True
                        else:
                            # print('242: find a gps in betw,
                            # which is far away with phone stay, contradic with a stay (with phone obsv)')
                            temp = user[d][phonestay_left][6:]
                            user[d][phonestay_left][6:] = [-1, -1, -1, -1, -1, -1]  # phone disappear
                            # user[d][phonestay_left].extend(temp)
                            user[d][phonestay_left].append('checked')
                            # del user[d][phonestay_left]  # phone disappear
                            flag_changed = True
                else:  # if they are two consecutive traces
                    # two consecutive observation
                    # if phonestay_left != 0 and phonestay_right < len(user[d]) - 1:
                    # ignore if they are at the beginning or the end of traj
                    prev_gps = next_gps = 0  # find prevous and next gps
                    found_prev_gps = False
                    found_next_gps = False
                    for prev in range(phonestay_left - 1, -1, -1):
                        # if int(user[d][prev][5]) <= spat_cell_split: ########## changed on 04282018
                        if int(user[d][prev][5]) <= spat_cell_split and int(user[d][prev][9]) >= dur_constr:
                            prev_gps = prev
                            found_prev_gps = True
                            break
                    for nxt in range(phonestay_right + 1, len(user[d])):
                        if int(user[d][nxt][5]) <= spat_cell_split and int(user[d][nxt][9]) >= dur_constr:
                            next_gps = nxt
                            found_next_gps = True
                            break

                    if found_prev_gps and found_next_gps and user[d][prev_gps][6] == user[d][next_gps][6]:
                        # this is a phone stay within a gps stay
                        phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                       int(user[d][phonestay_right][5])])
                        gps_uncernt = int(user[d][prev_gps][8])
                        dist = 1000 * distance(user[d][prev_gps][6],
                                               user[d][prev_gps][7],
                                               user[d][phonestay_left][6],
                                               user[d][phonestay_left][7])
                        speed_dep = (dist - phone_uncernt - gps_uncernt) / \
                                    (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
                        speed_retn = (dist - phone_uncernt - gps_uncernt) / \
                                     (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
                        if (dist - phone_uncernt - gps_uncernt) > 0 \
                                and dist > 1000*spat_constr_gps and speed_dep < 200 and speed_retn < 200:
                            # print('1111: distance larger than acc, and can travel, add phone stay, shorten gps stay')
                            # leave phone stay there, we later update duration for the gps stay
                            user[d][phonestay_left].append('checked')
                            # those phone stay not removed have to be marked with 'checked'!
                            user[d][phonestay_right].append('checked')
                            user[d][phonestay_left][2] = 'addedphonestay'
                            user[d][phonestay_right][2] = 'addedphonestay'
                            flag_changed = True
                        else:  # merge into gps stay
                            # print('1112: distance less than acc, or cannot travel, merge into gps stay')
                            temp = user[d][phonestay_left][3:]
                            user[d][phonestay_left][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            temp = user[d][phonestay_right][3:]
                            user[d][phonestay_right][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_right][11] = temp[8]
                            # user[d][phonestay_right].extend(temp)
                            flag_changed = True
                    elif found_prev_gps and found_next_gps and user[d][prev_gps][6] != user[d][next_gps][6]:
                        phone_uncernt_l = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                             int(user[d][phonestay_right][5])])
                        gps_uncernt_l = int(user[d][prev_gps][8])
                        dist_l = 1000 * distance(user[d][prev_gps][6],
                                           user[d][prev_gps][7],
                                           user[d][phonestay_left][6],
                                           user[d][phonestay_left][7])
                        speed_dep = (dist_l - phone_uncernt_l - gps_uncernt_l) / \
                                    (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
                        phone_uncernt_r = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                             int(user[d][phonestay_right][5])])
                        gps_uncernt_r = int(user[d][next_gps][8])
                        dist_r = 1000 * distance(user[d][next_gps][6],
                                           user[d][next_gps][7],
                                           user[d][phonestay_right][6],
                                           user[d][phonestay_right][7])
                        speed_retn = (dist_r - phone_uncernt_r - gps_uncernt_r) / \
                                     (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
                        comb_l = 0 #revised on 03202019 to pick up one gps stay to combine with; if spatial conti with multi
                        comb_r = 0
                        if (dist_l - phone_uncernt_l - gps_uncernt_l) < 0 \
                                or dist_l < 1000*spat_constr_gps or speed_dep > 200:
                            comb_l = 1
                        if (dist_r - phone_uncernt_r - gps_uncernt_r) < 0 \
                                or dist_r < 1000 * spat_constr_gps or speed_retn > 200:
                            comb_r = 1
                        if comb_l*comb_r == 1:
                            if dist_l < dist_r:
                                comb_r = 0
                            else:
                                comb_l = 0
                        if comb_l:
                            temp = user[d][phonestay_left][3:]
                            user[d][phonestay_left][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            temp = user[d][phonestay_right][3:]
                            user[d][phonestay_right][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_right][11] = temp[8]
                            # user[d][phonestay_right].extend(temp)
                            flag_changed = True
                        elif comb_r:
                            temp = user[d][phonestay_left][3:]
                            user[d][phonestay_left][3:] = user[d][next_gps][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            temp = user[d][phonestay_right][3:]
                            user[d][phonestay_right][3:] = user[d][next_gps][3:]
                            user[d][phonestay_right][11] = temp[8]
                            # user[d][phonestay_right].extend(temp)
                            flag_changed = True
                        else:
                            user[d][phonestay_left].append('checked')
                            # those phone stay not removed have to be marked with 'checked'!
                            user[d][phonestay_right].append('checked')
                            user[d][phonestay_left][2] = 'addedphonestay'
                            user[d][phonestay_right][2] = 'addedphonestay'
                            flag_changed = True
                    elif found_prev_gps:  # a gps stay #right# before
                        # print('113: before phone stay, we have gps stay')
                        phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                       int(user[d][phonestay_right][5])])
                        gps_uncernt = int(user[d][prev_gps][8])
                        dist = 1000 * distance(user[d][prev_gps][6],
                                               user[d][prev_gps][7],
                                               user[d][phonestay_left][6],
                                               user[d][phonestay_left][7])
                        speed_dep = (dist - phone_uncernt - gps_uncernt) / \
                                    (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
                        if (dist - phone_uncernt - gps_uncernt) > 0 and dist > 1000*spat_constr_gps and speed_dep < 200:
                            # spatially seperate enough and can travel, add in gps
                            # print('1132: dist>low_acc, add phone stay')
                            # leave phone stay there
                            user[d][phonestay_left].append('checked')
                            user[d][phonestay_right].append('checked')
                            user[d][phonestay_left][2] = 'addedphonestay'
                            user[d][phonestay_right][2] = 'addedphonestay'
                            flag_changed = True
                        else:
                            # print('1131: low_acc > dist, merge with gps stay, meaning extend gps dur')
                            temp = user[d][phonestay_left][3:]
                            user[d][phonestay_left][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            temp = user[d][phonestay_right][3:]
                            user[d][phonestay_right][3:] = user[d][prev_gps][3:]
                            user[d][phonestay_right][11] = temp[8]
                            # user[d][phonestay_right].extend(temp)
                            flag_changed = True
                    elif found_next_gps:  # a gps stay #right# after
                        # print('112: after phone stay, we have gps stay')
                        phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
                                       int(user[d][phonestay_right][5])])
                        gps_uncernt = int(user[d][next_gps][8])
                        dist = 1000 * distance(user[d][next_gps][6],
                                               user[d][next_gps][7],
                                               user[d][phonestay_right][6],
                                               user[d][phonestay_right][7])
                        speed_retn = (dist - phone_uncernt - gps_uncernt) / \
                                     (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
                        if (dist - phone_uncernt - gps_uncernt) > 0 and dist > 1000*spat_constr_gps and speed_retn<200:
                            # spatially seperate enough and can travel, add in gps
                            # print('1122: dist>low_acc, add phone stay')
                            # leave phone stay there, we later update duration for the gps stay
                            user[d][phonestay_left].append('checked')
                            user[d][phonestay_right].append('checked')
                            user[d][phonestay_left][2] = 'addedphonestay'
                            user[d][phonestay_right][2] = 'addedphonestay'
                            flag_changed = True
                        else:# remain phone observ, but use gps location
                            # print('1121: low_acc > dist, merge with gps stay, meaning extend gps dur')
                            temp = user[d][phonestay_left][3:]
                            user[d][phonestay_left][3:] = user[d][next_gps][3:]
                            user[d][phonestay_left][11] = temp[8]
                            # user[d][phonestay_left].extend(temp)
                            temp = user[d][phonestay_right][3:]
                            user[d][phonestay_right][3:] = user[d][next_gps][3:]
                            user[d][phonestay_right][11] = temp[8]
                            # user[d][phonestay_right].extend(temp)
                            flag_changed = True
                    else:  # if don't match any case, just add it
                        # print('donot match any case, just add it (e.g., consecutive two phone stays)')
                        # leave phone stay there
                        user[d][phonestay_left].append('checked')
                        user[d][phonestay_right].append('checked')
                        user[d][phonestay_left][2] = 'addedphonestay'
                        user[d][phonestay_right][2] = 'addedphonestay'
                        flag_changed = True


        # user[d].extend(phone_list_check)
        for trace in phone_list_check:
            if trace[2] == 'addedphonestay':
                user[d].append(trace[:])
        # remove passingby cellular traces
        i = 0
        while i<len(user[d]):
            if user[d][i][5] == 99 and float(user[d][i][9]) < dur_constr:
                del user[d][i]
            else:
                i+=1
        # remove passing traces
        ## Flag_changed = True
        ## while (Flag_changed):
        ## Flag_changed = False
        # i = 0
        # while i < len(user[d]):
        #     if int(user[d][i][5]) > spat_cell_split and int(user[d][i][9]) < dur_constr:
        #         # Flag_changed = True
        #         del user[d][i]
        #     else:
        #         i += 1
        user[d] = sorted(user[d], key=itemgetter(0))
        # update duration
        i = 0
        j = i
        while i < len(user[d]):
            if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
                dur = str(int(user[d][j - 1][0]) - int(user[d][i][0]))
                for k in range(i, j, 1):
                    user[d][k][9] = dur
                break
            if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(
                    user[d]):
                j += 1
            else:
                dur = str(int(user[d][j - 1][0]) - int(user[d][i][0]))
                for k in range(i, j, 1):
                    user[d][k][9] = dur
                i = j
        for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
            if float(trace[6]) == -1: trace[9] = -1
        if len(user[d]) == 1: user[d][0][9] = -1
        # remove and add back; because phone stays are distroyed as multiple, should be combined as one
        i = 0
        while i < len(user[d]):
            if user[d][i][2] == 'addedphonestay':
                del user[d][i]
            else:
                i += 1
        # add back and sort
        for trace in phone_list_check:
            if trace[2] == 'addedphonestay':
                user[d].append(trace)

        # # combine phone stays that are close to gps stay
        # gpsstays = set()
        # phonestays = set()
        # for trace in user[d]:
        #     if int(trace[9])>dur_constr:
        #         if trace[2] != 'addedphonestay':
        #             gpsstays.add((trace[6],trace[7]))
        #         else:
        #             phonestays.add((trace[6], trace[7]))
        # gpsstays = list(gpsstays)
        # phonestays = list(phonestays)
        # replacelist = {}
        # for point_p in phonestays:
        #     for point_g in gpsstays:
        #         if 1000*distance(point_p[0],point_p[1],point_g[0],point_g[1]) <= 1000*spat_constr_gps:
        #             replacelist[(point_p[0],point_p[1])] = (point_g[0],point_g[1])
        #             break
        # for trace in user[d]:
        #     if int(trace[9]) > 5 * 60 and (trace[6],trace[7]) in replacelist:
        #         trace[6], trace[7] = replacelist[(trace[6],trace[7])][0], replacelist[(trace[6],trace[7])][1]

        user[d] = sorted(user[d], key=itemgetter(0))

        #  remove temp marks
        user[d]=[trace[:12] for trace in user[d]]

    #  oscillation
    #  modify grid
    for day in user.keys():
        for trace in user[day]:
            if float(trace[6]) == -1:
                found_stay = False
                if found_stay == False:
                    trace[6] = trace[3] + '000'  # in case do not have enough digits
                    trace[7] = trace[4] + '000'
                    digits = (trace[6].split('.'))[1]
                    digits = digits[:2] + str(int(digits[2]) / 2)
                    trace[6] = (trace[6].split('.'))[0] + '.' + digits
                    # trace[6] = trace[6][:5] + str(int(trace[6][5]) / 2)  # 49.950 to 49.952 220 meters
                    digits = (trace[7].split('.'))[1]
                    digits = digits[:2] + str(int(digits[2:4]) / 25)
                    trace[7] = (trace[7].split('.'))[0] + '.' + digits
                    # trace[7] = trace[7][:7] + str(int(trace[7][7:9]) / 25)  # -122.3400 to -122.3425  180 meters
    # added to address oscillation
    OscillationPairList = oscillation_h1_oscill(user, dur_constr)  # in format: [, [ping[0], ping[1], pong[0], pong[1]]]
    # find all pair[1]s in list, and replace it with pair[0]
    for pair in OscillationPairList:
        for d in user.keys():
            for trace in user[d]:
                if trace[6] == pair[2] and trace[7] == pair[3]:
                    trace[6], trace[7] = pair[0], pair[1]

    # update duration
    user = update_duration(user)

    #  end addressing oscillation
    #  those newly added stays should be combined with close stays
    user = cluster_incremental(user, spat_constr_gps, dur_constr)
    #  update duration
    user = update_duration(user)
    #  use only one record for one stay
    for d in user:
        i = 0
        while i < len(user[d]) - 1:
            if user[d][i + 1][6] == user[d][i][6] and user[d][i + 1][7] == user[d][i][7] \
                    and user[d][i + 1][9] == user[d][i][9] and int(user[d][i][9]) >= dur_constr:
                del user[d][i + 1]
            else:
                i += 1
    # mark stay
    staylist = set()  # get unique staylist
    for d in user.keys():
        for trace in user[d]:
            if float(trace[9]) >= dur_constr:
                staylist.add((trace[6], trace[7]))
            else:  # change back keep full trajectory: do not use center for those are not stays
                trace[6], trace[7], trace[8], trace[9] = -1, -1, -1, -1  # for non stay, do not give center
    staylist = list(staylist)
    for d in user.keys():
        for trace in user[d]:
            for i in range(len(staylist)):
                if trace[6] == staylist[i][0] and trace[7] == staylist[i][1]:
                    trace[10] = 'stay' + str(i)
                    break

    return user


# def combineGPSandPhoneStops(arg):
#     user_gps = arg[0]
#     user_cell = arg[1]
#     dur_constr = arg[2]
#     spat_constr_gps = arg[3]
#     spat_cell_split = arg[4]
#
#     user = user_gps
#     for d in user.keys():
#         if len(user_cell[d]):
#             user[d].extend(user_cell[d])
#             user[d] = sorted(user[d], key=itemgetter(0))
#
#     for d in user:
#         phone_index = [k for k in range(len(user[d])) if int(user[d][k][5]) > spat_cell_split]
#         if len(phone_index) == 0:  # if no phone trace
#             continue
#         for i in range(len(user[d])):
#             if int(user[d][i][5]) > spat_cell_split and int(user[d][i][9]) < dur_constr:  # passing phone observ
#                 user[d][i].append('checked')
#         # combine consecutive obsv on a phone stay into two observ
#         i = min(phone_index)  # i has to be a phone index
#         j = i + 1
#         while i < len(user[d]) - 1:
#             if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
#                 for k in range(i + 1, j - 1, 1):
#                     user[d][k] = []
#                 break
#             if int(user[d][j][5]) > spat_cell_split and user[d][j][6] == user[d][i][6] \
#                     and user[d][j][7] == user[d][i][7] and j < len(user[d]):
#                 j += 1
#             else:
#                 for k in range(i + 1, j - 1, 1):
#                     user[d][k] = []
#                 phone_index = [k for k in range(j, len(user[d])) if int(user[d][k][5]) > spat_cell_split]
#                 if len(phone_index) < 3:  # if no phone trace
#                     break
#                 i = min(phone_index)  ##i has to be a phone index
#                 j = i + 1
#         i = 0  # remove []
#         while i < len(user[d]):
#             if len(user[d][i]) == 0:
#                 del user[d][i]
#             else:
#                 i += 1
#         # adress phone stay one by one
#         flag_changed = True
#         phone_list_check = []
#         while (flag_changed):
#             # print('while........')
#             flag_changed = False
#             gps_list = []
#             phone_list = []
#             for i in range(len(user[d])):
#                 if int(user[d][i][5]) <= spat_cell_split:  # or user[d][i][2] == 'addedphonestay': #changed on 0428
#                     gps_list.append(user[d][i])
#                 else:
#                     phone_list.append(user[d][i])
#
#             # # update gps stay
#             # i = 0
#             # j = i
#             # while i < len(gps_list):
#             #     if j >= len(gps_list):  # a day ending with a stay, j goes beyond the last observation
#             #         dur = str(int(gps_list[j - 1][0]) - int(gps_list[i][0]))
#             #         for k in range(i, j, 1):
#             #             gps_list[k][9] = dur
#             #         break
#             #     if gps_list[j][6] == gps_list[i][6] and gps_list[j][7] == gps_list[i][7] and j < len(
#             #             gps_list):
#             #         j += 1
#             #     else:
#             #         dur = str(int(gps_list[j - 1][0]) - int(gps_list[i][0]))
#             #         for k in range(i, j, 1):
#             #             gps_list[k][9] = dur
#             #         i = j
#             # for trace in gps_list:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
#             #     if int(trace[6]) == -1: trace[9] = -1
#             # if len(gps_list) == 1: gps_list[0][9] = -1
#
#             phone_list.extend(phone_list_check)
#             # when updating duration for phone stay, we have to put back passing obs
#             phone_list = sorted(phone_list, key=itemgetter(0))
#             # update phone stay
#             i = 0
#             j = i
#             while i < len(phone_list):
#                 if j >= len(phone_list):  # a day ending with a stay, j goes beyond the last observation
#                     dur = str(int(phone_list[j - 1][0]) - int(phone_list[i][0]))
#                     for k in range(i, j, 1):
#                         if int(phone_list[k][9]) >= dur_constr:
#                             # we don't want to change a pssing into a stay; as  we have not process the combine this stay
#                             # this is possible when a stay that prevents two passing is mergeed into gps as gps points
#                             phone_list[k][9] = dur
#                     break
#                 if phone_list[j][6] == phone_list[i][6] and phone_list[j][7] == phone_list[i][7] and j < len(
#                         phone_list):
#                     j += 1
#                 else:
#                     dur = str(int(phone_list[j - 1][0]) - int(phone_list[i][0]))
#                     for k in range(i, j, 1):
#                         if int(phone_list[k][9]) >= dur_constr:
#                             phone_list[k][9] = dur
#                     i = j
#             for trace in phone_list:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
#                 if float(trace[6]) == -1: trace[9] = -1
#             if len(phone_list) == 1: phone_list[0][9] = -1
#
#             # update check lable
#             for i in range(len(phone_list)):
#                 if int(phone_list[i][5]) > spat_cell_split and int(phone_list[i][9]) < dur_constr \
#                         and phone_list[i][-1] != 'checked':
#                     # passing phone observ
#                     phone_list[i].append('checked')
#
#             # put those not checked together with gps
#             user[d] = gps_list
#             phone_list_check = []
#             for i in range(len(phone_list)):
#                 if phone_list[i][-1] == 'checked':
#                     phone_list_check.append(phone_list[i])
#                 else:
#                     user[d].append(phone_list[i])
#             user[d] = sorted(user[d], key=itemgetter(0))
#
#             # find a stay which is not checked
#             flag_phonestay_notchecked = False
#             phonestay_left = -1
#             phonestay_right = -1
#             for i in range(len(user[d])):
#                 if int(user[d][i][5]) > spat_cell_split \
#                         and int(user[d][i][9]) >= dur_constr and user[d][i][-1] != 'checked':
#                     phonestay_left = phonestay_right
#                     phonestay_right = i
#                 if phonestay_left != -1 and phonestay_right != -1 and user[d][phonestay_left][9] == \
#                         user[d][phonestay_right][9]:
#                     flag_phonestay_notchecked = True
#                     break
#             # print('phonestay_left, phonestay_right', phonestay_left, phonestay_right, len(user[d]))
#             #
#             if flag_phonestay_notchecked == False or len(phone_list) == 0:  # if all phone observation are checked, end
#                 break
#
#             # if they are not two consecutive observation
#             if phonestay_right != phonestay_left + 1:  # attention: only phonestay_left is addressed
#                 # print('2: not consecutive two observations')
#                 # if phonestay_left == 0: #######
#                 #     pass
#                 # elif user[d][phonestay_left - 1][6] == user[d][phonestay_left + 1][6]:
#                 #     print('22: phonestay_left intersect a gps stay')
#                 #     # we don't want phone interupt a gps stay
#                 #     temp = user[d][phonestay_left]
#                 #     user[d][phonestay_left][3:] = user[d][phonestay_left + 1][3:]
#                 #     user[d][phonestay_left].extend(temp[3:])
#                 #     flag_changed = True
#                 if any([int(user[d][j][9]) >= dur_constr for j in range(phonestay_left + 1, phonestay_right, 1)]):
#                     # found a gps stay in betw #can be merge with 22
#                     # print('23: found a gps stay in betw, just use one gps stay trade one phone stay')
#                     temp = user[d][phonestay_left][6:]
#                     user[d][phonestay_left][6:] = [-1, -1, -1, -1, -1, -1]  # phone disappear
#                     # user[d][phonestay_left].extend(temp)
#                     user[d][phonestay_left].append('checked')
#                     # del user[d][phonestay_left]  # phone disappear
#                     flag_changed = True
#                 else:  # find close gps
#                     # print('24: do not found a gps stay in betw')
#                     phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                          int(user[d][phonestay_right][5])])
#                     if all([(phone_uncernt + int(user[d][j][5])) > 1000 * distance(user[d][j][3], user[d][j][4],
#                                                                                    user[d][phonestay_left][6],
#                                                                                    user[d][phonestay_left][7])
#                             for j in range(phonestay_left + 1, phonestay_right, 1)]):
#                         # total uncerty larger than distance
#                         # this case should be rare, as those close gps may be clustered
#                         # print('241: all gps falling betw are close with phone stay')
#                         temp = user[d][phonestay_left][3:]  # copy neighbor gps
#                         user[d][phonestay_left][3:] = user[d][phonestay_left + 1][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         flag_changed = True
#                     else:
#                         # print('242: find a gps in betw,
#                         # which is far away with phone stay, contradic with a stay (with phone obsv)')
#                         temp = user[d][phonestay_left][6:]
#                         user[d][phonestay_left][6:] = [-1, -1, -1, -1, -1, -1]  # phone disappear
#                         # user[d][phonestay_left].extend(temp)
#                         user[d][phonestay_left].append('checked')
#                         # del user[d][phonestay_left]  # phone disappear
#                         flag_changed = True
#             else:  # if they are two consecutive traces
#                 # print('two consecutive observation')
#                 # if phonestay_left != 0 and phonestay_right < len(user[d]) - 1:
#                 # ignore if they are at the beginning or the end of traj
#                 prev_gps = next_gps = 0  # find prevous and next gps
#                 found_prev_gps = False
#                 found_next_gps = False
#                 for prev in range(phonestay_left - 1, -1, -1):
#                     # if int(user[d][prev][5]) <= spat_cell_split: ########## changed on 0428
#                     if int(user[d][prev][5]) <= spat_cell_split and int(user[d][prev][9]) >= dur_constr:
#                         prev_gps = prev
#                         found_prev_gps = True
#                         break
#                 for nxt in range(phonestay_right + 1, len(user[d])):
#                     if int(user[d][nxt][5]) <= spat_cell_split and int(user[d][nxt][9]) >= dur_constr:
#                         next_gps = nxt
#                         found_next_gps = True
#                         break
#
#                 if found_prev_gps and found_next_gps and user[d][prev_gps][6] == user[d][next_gps][6]:
#                     # print('111: this is a phone stay within a gps stay')
#                     # this is a phone stay within a gps stay
#                     phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                          int(user[d][phonestay_right][5])])
#                     gps_uncernt = int(user[d][prev_gps][8])
#                     dist = 1000 * distance(user[d][prev_gps][6],
#                                            user[d][prev_gps][7],
#                                            user[d][phonestay_left][6],
#                                            user[d][phonestay_left][7])
#                     speed_dep = (dist - phone_uncernt - gps_uncernt) / \
#                                 (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
#                     speed_retn = (dist - phone_uncernt - gps_uncernt) / \
#                                  (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
#                     if (dist - phone_uncernt - gps_uncernt) > 0 \
#                             and dist > 1000 * spat_constr_gps and speed_dep < 200 and speed_retn < 200:
#                         # print('1111: distance larger than acc, and can travel, add phone stay, shorten gps stay')
#                         # leave phone stay there, we later update duration for the gps stay
#                         user[d][phonestay_left].append('checked')
#                         # those phone stay not removed have to be marked with 'checked'!
#                         user[d][phonestay_right].append('checked')
#                         user[d][phonestay_left][2] = 'addedphonestay'
#                         user[d][phonestay_right][2] = 'addedphonestay'
#                         flag_changed = True
#                     else:  # merge into gps stay
#                         # print('1112: distance less than acc, or cannot travel, merge into gps stay')
#                         temp = user[d][phonestay_left][3:]
#                         user[d][phonestay_left][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         temp = user[d][phonestay_right][3:]
#                         user[d][phonestay_right][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_right][11] = temp[8]
#                         # user[d][phonestay_right].extend(temp)
#                         flag_changed = True
#                 if found_prev_gps and found_next_gps and user[d][prev_gps][6] != user[d][next_gps][6]:
#                     phone_uncernt_l = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                            int(user[d][phonestay_right][5])])
#                     gps_uncernt_l = int(user[d][prev_gps][8])
#                     dist_l = 1000 * distance(user[d][prev_gps][6],
#                                              user[d][prev_gps][7],
#                                              user[d][phonestay_left][6],
#                                              user[d][phonestay_left][7])
#                     speed_dep = (dist_l - phone_uncernt_l - gps_uncernt_l) / \
#                                 (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
#                     phone_uncernt_r = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                            int(user[d][phonestay_right][5])])
#                     gps_uncernt_r = int(user[d][next_gps][8])
#                     dist_r = 1000 * distance(user[d][next_gps][6],
#                                              user[d][next_gps][7],
#                                              user[d][phonestay_right][6],
#                                              user[d][phonestay_right][7])
#                     speed_retn = (dist_r - phone_uncernt_r - gps_uncernt_r) / \
#                                  (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
#                     comb_l = 0  # revised on 03202019 to pick up one gps stay to combine with; if spatial conti with multi
#                     comb_r = 0
#                     if (dist_l - phone_uncernt_l - gps_uncernt_l) < 0 \
#                             or dist_l < 1000 * spat_constr_gps or speed_dep > 200:
#                         comb_l = 1
#                     if (dist_r - phone_uncernt_r - gps_uncernt_r) < 0 \
#                             or dist_r < 1000 * spat_constr_gps or speed_retn > 200:
#                         comb_r = 1
#                     if comb_l * comb_r == 1:
#                         if dist_l < dist_r:
#                             comb_r = 0
#                         else:
#                             comb_l = 0
#                     if comb_l:
#                         temp = user[d][phonestay_left][3:]
#                         user[d][phonestay_left][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         temp = user[d][phonestay_right][3:]
#                         user[d][phonestay_right][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_right][11] = temp[8]
#                         # user[d][phonestay_right].extend(temp)
#                         flag_changed = True
#                     elif comb_r:
#                         temp = user[d][phonestay_left][3:]
#                         user[d][phonestay_left][3:] = user[d][next_gps][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         temp = user[d][phonestay_right][3:]
#                         user[d][phonestay_right][3:] = user[d][next_gps][3:]
#                         user[d][phonestay_right][11] = temp[8]
#                         # user[d][phonestay_right].extend(temp)
#                         flag_changed = True
#                     else:
#                         user[d][phonestay_left].append('checked')
#                         # those phone stay not removed have to be marked with 'checked'!
#                         user[d][phonestay_right].append('checked')
#                         user[d][phonestay_left][2] = 'addedphonestay'
#                         user[d][phonestay_right][2] = 'addedphonestay'
#                         flag_changed = True
#                 elif found_prev_gps:  # a gps stay #right# before
#                     # print('113: before phone stay, we have gps stay')
#                     phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                          int(user[d][phonestay_right][5])])
#                     gps_uncernt = int(user[d][prev_gps][8])
#                     dist = 1000 * distance(user[d][prev_gps][6],
#                                            user[d][prev_gps][7],
#                                            user[d][phonestay_left][6],
#                                            user[d][phonestay_left][7])
#                     speed_dep = (dist - phone_uncernt - gps_uncernt) / \
#                                 (int(user[d][phonestay_left][0]) - int(user[d][prev_gps][0])) * 3.6
#                     if (dist - phone_uncernt - gps_uncernt) > 0 and dist > 1000 * spat_constr_gps and speed_dep < 200:
#                         # spatially seperate enough and can travel, add in gps
#                         # print('1132: dist>low_acc, add phone stay')
#                         # leave phone stay there
#                         user[d][phonestay_left].append('checked')
#                         user[d][phonestay_right].append('checked')
#                         user[d][phonestay_left][2] = 'addedphonestay'
#                         user[d][phonestay_right][2] = 'addedphonestay'
#                         flag_changed = True
#                     else:
#                         # print('1131: low_acc > dist, merge with gps stay, meaning extend gps dur')
#                         temp = user[d][phonestay_left][3:]
#                         user[d][phonestay_left][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         temp = user[d][phonestay_right][3:]
#                         user[d][phonestay_right][3:] = user[d][prev_gps][3:]
#                         user[d][phonestay_right][11] = temp[8]
#                         # user[d][phonestay_right].extend(temp)
#                         flag_changed = True
#                 elif found_next_gps:  # a gps stay #right# after
#                     # print('112: after phone stay, we have gps stay')
#                     phone_uncernt = max([int(user[d][phonestay_left][8]), int(user[d][phonestay_left][5]),
#                                          int(user[d][phonestay_right][5])])
#                     gps_uncernt = int(user[d][next_gps][8])
#                     dist = 1000 * distance(user[d][next_gps][6],
#                                            user[d][next_gps][7],
#                                            user[d][phonestay_right][6],
#                                            user[d][phonestay_right][7])
#                     speed_retn = (dist - phone_uncernt - gps_uncernt) / \
#                                  (int(user[d][next_gps][0]) - int(user[d][phonestay_right][0])) * 3.6
#                     if (dist - phone_uncernt - gps_uncernt) > 0 and dist > 1000 * spat_constr_gps and speed_retn < 200:
#                         # spatially seperate enough and can travel, add in gps
#                         # print('1122: dist>low_acc, add phone stay')
#                         # leave phone stay there, we later update duration for the gps stay
#                         user[d][phonestay_left].append('checked')
#                         user[d][phonestay_right].append('checked')
#                         user[d][phonestay_left][2] = 'addedphonestay'
#                         user[d][phonestay_right][2] = 'addedphonestay'
#                         flag_changed = True
#                     else:  # remain phone observ, but use gps location
#                         # print('1121: low_acc > dist, merge with gps stay, meaning extend gps dur')
#                         temp = user[d][phonestay_left][3:]
#                         user[d][phonestay_left][3:] = user[d][next_gps][3:]
#                         user[d][phonestay_left][11] = temp[8]
#                         # user[d][phonestay_left].extend(temp)
#                         temp = user[d][phonestay_right][3:]
#                         user[d][phonestay_right][3:] = user[d][next_gps][3:]
#                         user[d][phonestay_right][11] = temp[8]
#                         # user[d][phonestay_right].extend(temp)
#                         flag_changed = True
#                 else:  # if don't match any case, just add it
#                     # print('donot match any case, just add it (e.g., consecutive two phone stays)')
#                     # leave phone stay there
#                     user[d][phonestay_left].append('checked')
#                     user[d][phonestay_right].append('checked')
#                     user[d][phonestay_left][2] = 'addedphonestay'
#                     user[d][phonestay_right][2] = 'addedphonestay'
#                     flag_changed = True
#
#         # user[d].extend(phone_list_check)
#         for trace in phone_list_check:
#             if trace[2] == 'addedphonestay':
#                 user[d].append(trace[:])
#
#         # remove passing traces
#         ## Flag_changed = True
#         ## while (Flag_changed):
#         ## Flag_changed = False
#         # i = 0
#         # while i < len(user[d]):
#         #     if int(user[d][i][5]) > spat_cell_split and int(user[d][i][9]) < dur_constr:
#         #         # Flag_changed = True
#         #         del user[d][i]
#         #     else:
#         #         i += 1
#         user[d] = sorted(user[d], key=itemgetter(0))
#         # update duration
#         i = 0
#         j = i
#         while i < len(user[d]):
#             if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
#                 dur = str(int(user[d][j - 1][0]) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 break
#             if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(
#                     user[d]):
#                 j += 1
#             else:
#                 dur = str(int(user[d][j - 1][0]) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 i = j
#         for trace in user[d]:  # those trace with gps as -1,-1 (not clustered) should not assign a duration
#             if float(trace[6]) == -1: trace[9] = -1
#         if len(user[d]) == 1: user[d][0][9] = -1
#         # remove and add back; because phone stays are distroyed as multiple, should be combined as one
#         i = 0
#         while i < len(user[d]):
#             if user[d][i][2] == 'addedphonestay':
#                 del user[d][i]
#             else:
#                 i += 1
#         # add back and sort
#         for trace in phone_list_check:
#             if trace[2] == 'addedphonestay':
#                 user[d].append(trace)
#
#         # # combine phone stays that are close to gps stay
#         # gpsstays = set()
#         # phonestays = set()
#         # for trace in user[d]:
#         #     if int(trace[9])>dur_constr:
#         #         if trace[2] != 'addedphonestay':
#         #             gpsstays.add((trace[6],trace[7]))
#         #         else:
#         #             phonestays.add((trace[6], trace[7]))
#         # gpsstays = list(gpsstays)
#         # phonestays = list(phonestays)
#         # replacelist = {}
#         # for point_p in phonestays:
#         #     for point_g in gpsstays:
#         #         if 1000*distance(point_p[0],point_p[1],point_g[0],point_g[1]) <= 1000*spat_constr_gps:
#         #             replacelist[(point_p[0],point_p[1])] = (point_g[0],point_g[1])
#         #             break
#         # for trace in user[d]:
#         #     if int(trace[9]) > 5 * 60 and (trace[6],trace[7]) in replacelist:
#         #         trace[6], trace[7] = replacelist[(trace[6],trace[7])][0], replacelist[(trace[6],trace[7])][1]
#
#         user[d] = sorted(user[d], key=itemgetter(0))
#
#         #  remove temp marks
#         user[d] = [trace[:12] for trace in user[d]]
#
#     #  oscillation
#     #  modify grid
#     for day in user.keys():
#         for trace in user[day]:
#             if float(trace[6]) == -1:
#                 found_stay = False
#                 if found_stay == False:
#                     trace[6] = trace[3] + '000'  # in case do not have enough digits
#                     trace[7] = trace[4] + '000'
#                     digits = (trace[6].split('.'))[1]
#                     digits = digits[:2] + str(int(digits[2]) / 2)
#                     trace[6] = (trace[6].split('.'))[0] + '.' + digits
#                     # trace[6] = trace[6][:5] + str(int(trace[6][5]) / 2)  # 49.950 to 49.952 220 meters
#                     digits = (trace[7].split('.'))[1]
#                     digits = digits[:2] + str(int(digits[2:4]) / 25)
#                     trace[7] = (trace[7].split('.'))[0] + '.' + digits
#                     # trace[7] = trace[7][:7] + str(int(trace[7][7:9]) / 25)  # -122.3400 to -122.3425  180 meters
#     # added to address oscillation
#     OscillationPairList = oscillation_h1_oscill(user, dur_constr)  # in format: [, [ping[0], ping[1], pong[0], pong[1]]]
#     # find all pair[1]s in list, and replace it with pair[0]
#     for pair in OscillationPairList:
#         for d in user.keys():
#             for trace in user[d]:
#                 if trace[6] == pair[2] and trace[7] == pair[3]:
#                     trace[6], trace[7] = pair[0], pair[1]
#
#     # update duration
#     for d in user.keys():
#         for trace in user[d]: trace[9] = -1  # clear needed! #modify grid
#         i = 0
#         j = i
#         while i < len(user[d]):
#             if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
#                 dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 break
#             if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(user[d]):
#                 j += 1
#             else:
#                 dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 i = j
#     #  end addressing oscillation
#     #  those newly added stays should be combined with close stays
#     user = cluster_incremental(user, spat_constr_gps, dur_constr)
#     #  update duration
#     for d in user.keys():
#         for trace in user[d]: trace[9] = -1  # clear needed! #modify grid
#         i = 0
#         j = i
#         while i < len(user[d]):
#             if j >= len(user[d]):  # a day ending with a stay, j goes beyond the last observation
#                 dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 break
#             if user[d][j][6] == user[d][i][6] and user[d][j][7] == user[d][i][7] and j < len(user[d]):
#                 j += 1
#             else:
#                 dur = str(int(user[d][j - 1][0]) + max(0, int(user[d][j - 1][9])) - int(user[d][i][0]))
#                 for k in range(i, j, 1):
#                     user[d][k][9] = dur
#                 i = j
#     #  use only one record for one stay
#     for d in user:
#         i = 0
#         while i < len(user[d]) - 1:
#             if user[d][i + 1][6] == user[d][i][6] and user[d][i + 1][7] == user[d][i][7] \
#                     and user[d][i + 1][9] == user[d][i][9] and int(user[d][i][9]) >= dur_constr:
#                 del user[d][i + 1]
#             else:
#                 i += 1
#     # mark stay
#     staylist = set()  # get unique staylist
#     for d in user.keys():
#         for trace in user[d]:
#             if float(trace[9]) >= dur_constr:
#                 staylist.add((trace[6], trace[7]))
#             else:  # change back keep full trajectory: do not use center for those are not stays
#                 trace[6], trace[7], trace[8], trace[9] = -1, -1, -1, -1  # for non stay, do not give center
#     staylist = list(staylist)
#     for d in user.keys():
#         for trace in user[d]:
#             for i in range(len(staylist)):
#                 if trace[6] == staylist[i][0] and trace[7] == staylist[i][1]:
#                     trace[10] = 'stay' + str(i)
#                     break
#
#     return user


def func_identify_trip_ends(arg):
    name = arg[0]
    user = arg[1]
    dur_constr = arg[2]
    spat_constr_gps = arg[3]
    spat_constr_cell = arg[4]
    spat_cell_split = arg[5]
    workdir = arg[6]

    # print('processing name: ', name)
    try:
        user_gps = {}
        user_cell = {}
        for d in user.keys():
            user_gps[d] = []
            user_cell[d] = []
            for trace in user[d]:
                if int(trace[5]) <= spat_cell_split:
                    user_gps[d].append(trace)
                else:
                    user_cell[d].append(trace)

        user_gps = clusterGPS((user_gps, dur_constr, spat_constr_gps))
        user_cell = clusterPhone((user_cell, dur_constr, spat_constr_cell))
        user = combineGPSandPhoneStops((user_gps, user_cell, dur_constr, spat_constr_gps, spat_cell_split))
    except:
        err_name = workdir + 'log_not_processed_names.csv'
        f = open(err_name, 'ab')
        writeCSV = csv.writer(f, delimiter='\t')
        writeCSV.writerow([name])
        f.close()

    filenamewrite = workdir + str(current_process())[13:21] + '.csv'
    with lock:
        f = open(filenamewrite, 'ab')
        writeCSV = csv.writer(f, delimiter='\t')
        for day in sorted(user.keys()):
            if len(user[day]):
                for trace in user[day]:
                    trace[1] = name
                    trace[6], trace[7] = round(float(trace[6]), 7), round(float(trace[7]), 7)
                    writeCSV.writerow(trace)
        f.close()
        # print('processed name: ',name)
    user_gps = None
    user_cell = None
    user = None


if __name__ == '__main__':

    day_list = ['18110'+str(i) if i < 10 else '1811'+str(i) for i in range(1,31)]
    day_list.extend(['18120'+str(i) if i < 10 else '1812'+str(i) for i in range(1,32)])
    day_list.extend(['19010' + str(i) if i < 10 else '1901' + str(i) for i in range(1, 32)])
    day_list.extend(['19020' + str(i) if i < 10 else '1902' + str(i) for i in range(1, 29)])

    # day_list = ['18010'+str(i) if i < 10 else '1801'+str(i) for i in range(1,32)]
    # day_list.extend(['18020'+str(i) if i < 10 else '1802'+str(i) for i in range(1,29)])
    # day_list.extend(['18030' + str(i) if i < 10 else '1803' + str(i) for i in range(1, 32)])

    cpu_useno = 4  # cpu_count() - 2
    l = Lock()
    pool = Pool(cpu_useno, initializer=init, initargs=(l,))

    partList = ['47']
    # partList.extend(['0'+str(i) if i < 10 else str(i) for i in range(48,57)])
    for part_num in partList:
        # part_num = '00'
        print('partnum: ', part_num)
        with open(infile_workdir+'part201811_'+ part_num + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
                # next(readCSV)
            usernamelist_1 = set([row[1] for row in readCSV])
        # # deal with not completed
        # with open(outfile_workdir+'trip_identified_part201811_'+ part_num + 'notcomplete.csv') as csvfile:
        #     readCSV = csv.reader(csvfile, delimiter='\t')
        #     usernamelist_2 = set([row[1] for row in readCSV])
        # usernamelist_1 = usernamelist_1.difference(usernamelist_2)

        usernamelist = []
        counti = 0
        namebulk = []
        for name in list(usernamelist_1):
            if (counti < user_num_in_mem):
                namebulk.append(name)
                counti += 1
            else:
                usernamelist.append(namebulk)
                namebulk = []
                namebulk.append(name)
                counti = 1
        usernamelist.append(namebulk)  # the last one which is smaller than 100000

        usernamelist_1 = None
        usernamelist_2 = None
        print(len(usernamelist))
        print(sum([len(bulk) for bulk in usernamelist]))

        while (len(usernamelist)):
            bulkname = usernamelist.pop()
            print("Start processing bulk: ", len(usernamelist)+1,
                  ' at time: ', time.strftime("%m%d-%H:%M"),' memory: ', psutil.virtual_memory().percent)

            UserList = {}
            for name in bulkname:
                UserList[name] = {day: [] for day in day_list}

            with open(infile_workdir+'part201811_'+ part_num + '.csv') as readfile:
                readCSV = csv.reader(readfile, delimiter='\t')
                # next(readCSV)
                for row in readCSV:
                    name = row[1]
                    if name in UserList:
                        if row[6][:6] not in day_list: continue
                        row[1] = None
                        row.extend([-1, -1, -1, -1, -1])
                        row[6], row[11] = row[11], row[6]
                        # day = int(row[11][:2])
                        if int(row[11][6:8]) < 3: # effective day
                            whichday = day_list.index(row[11][:6])-1 # go to previous day
                            if whichday < 0: continue
                            UserList[name][day_list[whichday]].append(row)
                        else:
                            UserList[name][row[11][:6]].append(row)

            sortednames = {} # sort names and tackle large names first
            for name in UserList:
                for day in UserList[name]: # debug a data issue
                    i = 0
                    while i < len(UserList[name][day]):
                        if '.' not in UserList[name][day][i][3] or '.' not in UserList[name][day][i][4]:
                            del UserList[name][day][i]
                        else:
                            i += 1
                sortednames[name] = sum([len(UserList[name][day]) for day in UserList[name]]) # process easy ones first
            sortednames = sorted(sortednames.items(), key=lambda kv: kv[1], reverse=True)
            sortednames = [item[0] for item in sortednames]

            print ("End reading; start calculating...")

            tasks = [pool.apply_async(func_identify_trip_ends, (task,)) for task in [(name, UserList[name], dur_constr, spat_constr_gps, spat_constr_cell, spat_cell_split, outfile_workdir) for name in sortednames]]
            finishit = [t.get() for t in tasks]
            # pool.map(func_identify_trip_ends, [(name, UserList[name], dur_constr, spat_constr_gps, spat_constr_cell,
            #                                     spat_cell_split, outfile_workdir) for name in sortednames]) # process easy ones first

            print ('End processing bulk: ', len(usernamelist)+1,' memory: ', psutil.virtual_memory().percent)

        print('start collecting together...')
        collect_filenames = glob.glob(outfile_workdir + 'Worker-*.csv')
        target_filename = outfile_workdir+'trip_identified_part201811_'+str(part_num)+'complement.csv'
        with open(target_filename, 'wb') as f_to:
            for filename in collect_filenames:
                with open(filename, 'rb') as f_from:
                    shutil.copyfileobj(f_from, f_to)
                os.remove(filename)
    pool.close()
    pool.join()

    print ('ALL Finished! Thank you!',' at time: ', time.strftime("%m%d-%H:%M"))



