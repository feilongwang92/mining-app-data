
import csv

if __name__ == '__main__':
    day_list = ['18110'+str(i) if i < 10 else '1811'+str(i) for i in range(1,31)]
    day_list.extend(['18120'+str(i) if i < 10 else '1812'+str(i) for i in range(1,32)])
    day_list.extend(['19010' + str(i) if i < 10 else '1901' + str(i) for i in range(1, 32)])
    day_list.extend(['19020' + str(i) if i < 10 else '1902' + str(i) for i in range(1, 29)])

    f = open('daily num users observations.csv', 'ab')
    writeCSV = csv.writer(f, delimiter='\t')
    row2write = ['part_num']
    row2write.extend(day_list)
    row2write.extend(day_list)
    writeCSV.writerow(row2write)

    for part_num in ['0'+str(i) if i < 10 else str(i) for i in range(57)]:
        with open('E:\\cuebiq_psrc_2019\\sorted\\part201811_'+ part_num + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            # next(readCSV)
            usernamelist_1 = list(set([row[1] for row in readCSV]))
        UserList = {name:set() for name in usernamelist_1}
        day_list_obs = {day:0 for day in day_list}

        with open('E:\\cuebiq_psrc_2019\\sorted\\part201811_' + part_num + '.csv') as readfile:
            readCSV = csv.reader(readfile, delimiter='\t')
            # next(readCSV)
            for row in readCSV:
                name = row[1]
                if row[6][:6] in day_list:
                    UserList[name].add(row[6][:6])
                    day_list_obs[row[6][:6]] += 1

        day_list_user = {day:0 for day in day_list}
        for name in UserList:
            for day in list(UserList[name]):
                day_list_user[day] += 1

        f = open('daily num users observations.csv', 'ab')
        writeCSV = csv.writer(f, delimiter='\t')
        row2write = [part_num]
        for day in sorted(day_list_user.keys()): row2write.append(day_list_user[day])
        for day in sorted(day_list_obs.keys()): row2write.append(day_list_obs[day])
        writeCSV.writerow(row2write)
    f.close()