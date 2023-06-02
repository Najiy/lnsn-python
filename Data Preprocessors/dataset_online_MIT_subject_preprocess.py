import pandas as pd
import numpy as np
import pprint as pp
import json
from datetime import datetime as datetime

folder = 'subject1'

def read_sensors(path=f'../dataset online/MIT/{folder}/sensors.csv'):
    sensors = pd.read_csv(path)
    return sensors

def read_activities():
    activities_fp = open(f'../dataset online/MIT/{folder}/activities_data.csv', 'r+').readlines()
    sensors_fp = open(f'../dataset online/MIT/{folder}/sensors.csv', 'r+').readlines()

    sensors = {}

    for sensor in sensors_fp:
        sensor = sensor.replace('\n','').split(',')
        if sensor[0].isnumeric():
            sensors[sensor[0]] = f"{sensor[2]}"

    sensors = [x for x in sensors.keys()]
    sensors.sort()

    csv_header = sensors
    csv_header.insert(0, "time")

    activities_dict = {}
    timeline = {}

    mintime = -1
    maxtime = -1
    counter = 0
    while (counter != len(activities_fp)):
        header = activities_fp[counter].replace('\n','').split(',')
        sensorid = activities_fp[counter+1].replace('\n','')
        sensorname = activities_fp[counter+2].replace('\n','')
        starttimes = activities_fp[counter+3].replace('\n','')
        endtimes = activities_fp[counter+4].replace('\n','')

        sensorid = sensorid.split(',')
        starttimes = starttimes.replace('\n','').split(',')
        endtimes = endtimes.replace('\n','').split(',')

        sequence = []
        for i, t in enumerate(starttimes):
            try:
                start = int(datetime.strptime(f"{header[1]} {starttimes[i]}", "%m/%d/%Y %H:%M:%S").strftime('%s'))
                end = int(datetime.strptime(f"{header[1]} {endtimes[i]}", "%m/%d/%Y %H:%M:%S").strftime('%s'))

                if mintime == -1 or mintime > start:
                    mintime = start
                if maxtime == -1 or maxtime < end:
                    maxtime = end

                if start not in timeline:
                    timeline[start] = []
                if end not in timeline:
                    timeline[end] = []

                timeline[start].append((sensorid[i], 1))
                timeline[end].append((sensorid[i], 0))
                sequence.append([sensorid[i], start, end])
            except:
                pass
        
        counter += 5

        if header[0] not in activities_dict:
            activities_dict[header[0]] = []

        sequence.sort(key=lambda x: x[1])
        
        activities_dict[header[0]].append(sequence)

    csv_timeline = []

    for i in range(mintime, maxtime):
        if i in timeline:
            entry = ['' for x in sensors]
            entry[0] = str(i)
            for stup in timeline[i]:
                entry[sensors.index(stup[0])] = str(stup[1])
            
            csv_timeline.append(','.join(entry)+',\n')
    csv_timeline.insert(0, ','.join(sensors)+',\n')

    csv_meta = []
    csv_meta.append('sensor,records,elapsed,unix_oldest,unix_newest,oldest,newest,minimum,maximum,min2,max2,min3,max3,\n')
    for i in sensors:
        csv_meta.append(f'{i},,,,,,,,,0,1,,,\n')

    return csv_timeline, csv_meta, activities_dict

activities, meta, activities_dict = read_activities()

open(f'../dataset online/MIT/{folder}/{folder}.csv','w+').writelines(activities)
open(f'../dataset online/MIT/{folder}/{folder}_meta.csv','w+').writelines(meta)
with open(f'../dataset online/MIT/{folder}/{folder}_act_dict.json', 'w+', encoding='utf-8') as f:
    json.dump(activities_dict, f, ensure_ascii=False, indent=4)