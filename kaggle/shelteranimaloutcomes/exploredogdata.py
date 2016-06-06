import pandas as pd
import numpy as np
import matplotlib
from dog_grp_arr import *
import sys
import os
matplotlib.rcParams.update({'font.size': 12})

df = pd.read_csv('/Volumes/work/data/kaggle/shelteranimaloutcomes/train.csv',sep=',')
feature = 'Breed'

feature = 'Breed'

feature_values_dog = df.loc[df['AnimalType'] == 'Dog', feature]
outcome_dog = df.loc[df['AnimalType'] == 'Dog', 'OutcomeType']
outcome_dog = np.array(outcome_dog)

#unique outcomes:
unique_outcomes = np.unique(outcome_dog)

# Convert the breed string into group lists

group_values_dog = []

count = 0

not_found = []

for i in feature_values_dog:
    i = i.replace(' Shorthair','')
    i = i.replace(' Longhair','')
    i = i.replace(' Wirehair','')
    i = i.replace(' Rough','')
    i = i.replace(' Smooth Coat','')
    i = i.replace(' Smooth','')
    i = i.replace(' Black/Tan','')
    i = i.replace('Black/Tan ','')
    i = i.replace(' Flat Coat','')
    i = i.replace('Flat Coat ','')
    i = i.replace(' Coat','')

    groups = []
    if '/' in i:
        split_i = i.split('/')
        for j in split_i:
            if j[-3:] == 'Mix':
                breed = j[:-4]
                if breed in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == breed)[0]
                    groups.append(breeds_group[indx,1][0])
                    groups.append('Mix')
                elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]
                    groups.append(find_group[find_group != 'Unknown'])
                    groups.append('Mix')
                elif breed == 'Pit Bull':
                    groups.append('Pit Bull')
                    groups.append('Mix')
                elif 'Shepherd' in breed:
                    groups.append('Herding')
                    groups.append('Mix')
                else:
                    not_found.append(breed)
                    groups.append('Unknown')
                    groups.append('Mix')
            else:
                if j in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == j)[0]
                    groups.append(breeds_group[indx,1][0])
                elif np.any([s.lower() in j.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]
                    groups.append(find_group[find_group != 'Unknown'])
                elif j == 'Pit Bull':
                    groups.append('Pit Bull')
                elif 'Shepherd' in j:
                    groups.append('Herding')
                    groups.append('Mix')
                else:
                    not_found.append(j)
                    groups.append('Unknown')
    else:

        if i[-3:] == 'Mix':
            breed = i[:-4]
            if breed in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == breed)[0]
                groups.append(breeds_group[indx,1][0])
                groups.append('Mix')
            elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]
                groups.append(find_group[find_group != 'Unknown'])
                groups.append('Mix')
            elif breed == 'Pit Bull':
                groups.append('Pit Bull')
                groups.append('Mix')
            elif 'Shepherd' in breed:
                groups.append('Herding')
                groups.append('Mix')
            else:
                groups.append('Unknown')
                groups.append('Mix')
                not_found.append(breed)

        else:
            if i in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == i)[0]
                groups.append(breeds_group[indx,1][0])
            elif np.any([s.lower() in i.lower() for s in dog_groups]):
                find_group = [s if s.lower() in i.lower() else 'Unknown' for s in dog_groups]
                groups.append(find_group[find_group != 'Unknown'])
            elif i == 'Pit Bull':
                groups.append('Pit Bull')
            elif 'Shepherd' in i:
                groups.append('Herding')
                groups.append('Mix')
            else:
                groups.append('Unknown')
                not_found.append(i)
    group_values_dog.append(list(set(groups)))


not_f_unique,counts = np.unique(not_found,return_counts= True)

unique_groups, counts = np.unique(group_values_dog,return_counts=True)

groups = np.unique(np.append(dog_groups,['Mix','Pit Bull','Unknown']))
print "dog_groups ",groups," unique_outcomes ",len(unique_outcomes)






