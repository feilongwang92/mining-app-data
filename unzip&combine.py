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

###unzip and combine
import shutil
import os

filedaylist = os.listdir('E:\\cuebiq_psrc_2019\\original')
for fileday in filedaylist:
    print(fileday)
    with open('E:\\cuebiq_psrc_2019\\sorted\\unzip' + fileday[:8] + '.csv', 'wb') as wfd:
        for filename in os.listdir('E:\\cuebiq_psrc_2019\\original\\' + fileday):
            with gzip.open('E:\\cuebiq_psrc_2019\\original\\' + fileday + '\\' + filename) as fd:
                shutil.copyfileobj(fd, wfd)

# combine one user observations together
