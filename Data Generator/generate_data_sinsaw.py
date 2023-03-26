import pprint as pp
from copy import deepcopy



sin = open("../Dataset/old/dataset_sin_S10F10.csv")
sin = sin.readlines()
saw = open("../Dataset/old/dataset_saw_S10F10.csv")
saw = saw.readlines()



length = min(len(sin) - 1, len(saw) - 1)
print('min', length)

headers = sin[0].split(',')
headers.remove('\n')

sin.pop(0)
saw.pop(0)

total = 0



newarray = []
newarray.append(headers)

for t in range(length):
    sawentry = ['1' if x != '' else '' for x in saw[t].replace('\n','').split(',') ]
    sinentry = ['1' if x != '' else '' for x in sin[t].replace('\n','').split(',') ]
    entry = ['' for x in sawentry]

    for i, e in enumerate(sawentry):
        
        if sawentry[i] != '':
            entry[i] = sawentry[i]
            total += 1
            # print('after saw', entry)

        if sinentry[i] != '':
            entry[i] = sinentry[i]
            total += 1
            # print('after sine', entry)

    entry[0] = t
    newarray.append(entry)



content = []
for n in newarray:
    content.append(','.join([str(x) for x in n])+'\n')


output = open("../Dataset/old/dataset_sinsaw_S10F10.csv", 'w+')
output.writelines(content)