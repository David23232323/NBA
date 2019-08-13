import numpy as np 
import csv
import datetime 
import calendar 
#reads csv file and feature engineering 
import pandas as pd


def read_data(path): #Returns separately all the Y, X, and a Panda Dataframe of
                     #of Y and X values 
    with open(path, 'r') as f:
        data = csv.reader(f, delimiter=',') 
        #[(y, [feature vector]),...]
        X = []
        Y = []
        panda = []
        for i, line in enumerate(data):
            if i == 0: #first line is title
                continue 
            else:
                Y.append(int(line[0]))
                feature_vector = feature_engineer(line[1:])
                X.append(feature_vector)
                a = (np.array([int(line[0])]))
                b = feature_vector
                panda.append(np.concatenate((a,b)))
    return np.array(Y), np.array(X), np.array(panda) 



def feature_engineer(x): #feature engineers an feature vector 
    #bag of words of important words 
    bag = {"Champions": 0, "stephencurry30": 1, "warriors": 2, "#NBAFinals": 3, "record": 4, "Kawhi":5, "#NBAPlayoffs":6, "Finals":7, "victory":8, "Semifinals":9, "#NBAPlayoffs!":10, "advance":11, "career-high":12, "#NBABreakdown":13, "buzzer":14, "TRIPLE":15, "SLAM":16,  "horn": 17}

    feature_vector = []

    engagment = np.array([x[0]]) #engagment

    date, time, _ = x[1].split(" ")
    year, month, day = date.split("-")
    day_of_week_vector = np.zeros(7) #starts on monday
    day_of_week = datetime.date(int(year),int(month),int(day)).weekday() #0 for monday, 1 for tues...
    day_of_week_vector[day_of_week] = 1

    hours = np.zeros(24)
    hours[int(time[:2])] = 1

    medium = ["Photo", "Album", "Video"]
    medium_vector = np.zeros(3)
    medium_index = 2
    medium_vector[medium.index(x[medium_index])] = 1

    description_index = 3
    description_vector = generate_description_vector(x[description_index], bag)

    return np.concatenate((engagment, day_of_week_vector, hours, medium_vector,
        description_vector))


def generate_description_vector(sentence, bag): #creates a sub vector that  
                                                #gives information on the text
    description_vector = np.zeros(len(bag))
    for word in sentence.split(" "):
        if word in bag:
            description_vector[bag[word]] = 1 #using 0, 1 encoding to signify 
                                              #if a keyword is present 
    return description_vector




