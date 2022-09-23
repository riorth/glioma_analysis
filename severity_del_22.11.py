# Библиотеки
import os
from typing import TextIO, List, Dict, Any, Optional
import sys
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import ndarray, cov
from numpy.linalg import eig
from pandas import DataFrame
import statistics
from numpy import linalg as la
from scipy.spatial import distance
import sklearn
from scipy.spatial.distance import cdist
from sklearn import decomposition
from sklearn import preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import scipy
from scipy.linalg import eigh
import json
import sys
import csv
import math
#import rampy as rp
import peakutils
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from scipy.signal import medfilt
from scipy.signal import savgol_filter
#import openpyxl
f_covid = os.listdir('d:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\test_covid\\scan')
severe_level_dir='d:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\test_covid\\Baza1.dat'
dir_covid = 'd:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\test_covid\\scan\\'
f_healthy = os.listdir('d:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\test_covid\\healthy')
dir_healthy = 'd:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\test_covid\\healthy\\'
if '.DS_Store' in f_covid:
    f_covid.remove('.DS_Store')
if '.DS_Store' in f_healthy:
    f_healthy.remove('.DS_Store')

wavelength = []
wavelength2=[]
wavelength1=[]
spectr_covid_severe3 = []
spectr_covid_severe2=[]
spectr_covid_severe1=[]
spectr_healthy = []
people_c_severe3 = 0
people_c_severe2=0
people_c_severe1=0
numbers_c3 = 0
numbers_c2=0
numbers_c1=0
people_h = 0
numbers_h = 0
count_wavel = 0

#Чтение названий нужных файлов из документа
read_severity=pd.read_csv(severe_level_dir,sep='\t',encoding='latin-1')
severity=np.asarray(read_severity['severity'].astype(int))
File=np.asarray(read_severity['File'].astype(str))
patient_numbers=np.asarray(read_severity['patient'].astype(int))
df=pd.DataFrame(data=[patient_numbers,File,severity]).T

df=df.set_axis(['patient','File','severity'],axis=1,inplace=False)

#Разделение по тяжести заболевания
severe3_file_names=[]
severe2_file_names=[]
severe1_file_names=[]
patient=[[] for i in range(3)]
severe3=df.index[df['severity']==3].tolist()
severe2=df.index[df['severity']==2].tolist()
severe1=df.index[df['severity']==1].tolist()
for indeces in severe3:
    severe3_file_names.append(File[indeces])
    patient[2].append(patient_numbers[indeces])
for indecess in severe2:
    severe2_file_names.append(File[indecess])
    patient[1].append(patient_numbers[indecess])
for ind in severe1:
    severe1_file_names.append(File[ind])
    patient[0].append(patient_numbers[ind])

indeces_severe3=[]
indeces_severe2=[]
indeces_severe1=[]
for severe_name in severe3_file_names:
    for file_name in f_covid:
        if severe_name in file_name:
            indeces_severe3.append(f_covid.index(file_name))
for name in severe2_file_names:
    for file_name in f_covid:
        if name in file_name:
            indeces_severe2.append(f_covid.index(file_name))
for name in severe1_file_names:
    for file_name in f_covid:
        if name in file_name:
            indeces_severe1.append(f_covid.index(file_name))

#Чтение больных по тяжести заболевания и запись их в массивы
for index in indeces_severe3:
    for file in f_covid:
        x=f_covid.index(file)
        if x==index:
            read_covid_severe3=pd.read_csv(dir_covid+f_covid[index],sep='\t')
            wavel = np.asarray(read_covid_severe3['Wl'].astype(float))
            spectr_c = np.asarray(read_covid_severe3['OAD'].astype(float))
            spectr_covid_severe3.extend(spectr_c)
            wavelength.extend(wavel)
            people_c_severe3 += 1
            for data in spectr_c:
                numbers_c3 = numbers_c3 + 1

for index in indeces_severe2:
    for file in f_covid:
        y=f_covid.index(file)
        if y==index:
            read_covid2=pd.read_csv(dir_covid+f_covid[index],sep='\t')
            spectr_c = np.asarray(read_covid2['OAD'].astype(float))
            wavel2=np.asarray(read_covid2['Wl'].astype(float))
            spectr_covid_severe2.extend(spectr_c)
            wavelength2.extend(wavel2)
            people_c_severe2+= 1
            for data in spectr_c:
                numbers_c2 = numbers_c2 + 1

for index in indeces_severe1:
    for file in f_covid:
        z=f_covid.index(file)
        if z==index:
            read_covid1=pd.read_csv(dir_covid+f_covid[index],sep='\t')
            spectr_c = np.asarray(read_covid1['OAD'].astype(float))
            wavel1 = np.asarray(read_covid1['Wl'].astype(float))
            spectr_covid_severe1.extend(spectr_c)
            wavelength1.extend(wavel1)
            people_c_severe1+= 1
            for data in spectr_c:
                numbers_c1 = numbers_c1 + 1

#Формирование дф по тяжести заболевания
df_c_severe3 = pd.DataFrame(data=[spectr_covid_severe3, wavelength]).T
df_c_severe3 = df_c_severe3.set_axis(['Spectrum_c_severe3', 'Wavel'],
                                     axis=1, inplace=False)
df_c_severe2 = pd.DataFrame(data=[spectr_covid_severe2, wavelength2]).T
df_c_severe2 = df_c_severe2.set_axis(['Spectrum_c_severe2', 'Wavel'],
                                     axis=1, inplace=False)
df_c_severe1 = pd.DataFrame(data=[spectr_covid_severe1, wavelength1]).T
df_c_severe1 = df_c_severe1.set_axis(['Spectrum_c_severe1', 'Wavel'],
                                     axis=1, inplace=False)

#Чистка больных
df_del3 = df_c_severe3.loc[(df_c_severe3['Wavel'] >= 9.8) &
                           (df_c_severe3['Wavel'] <= 10.1)]
df_del2 = df_c_severe2.loc[(df_c_severe2['Wavel'] >= 9.8) &
                           (df_c_severe2['Wavel'] <= 10.1)]
df_del1 = df_c_severe1.loc[(df_c_severe1['Wavel'] >= 9.8) &
                           (df_c_severe1['Wavel'] <= 10.1)]
delete=[]
for wavel_del in wavelength:
    if (wavel_del >= 9.8) & (wavel_del <= 10.1):
        delete.append(wavel_del)
df_new_severe3 = df_c_severe3[~df_c_severe3['Wavel'].isin(delete)]
df_new_severe2=df_c_severe2[~df_c_severe2['Wavel'].isin(delete)]
df_new_severe1=df_c_severe1[~df_c_severe1['Wavel'].isin(delete)]
for i3 in df_new_severe3:
    numbers_c3=numbers_c3-1
numbers_c3=int(numbers_c3/people_c_severe3)
for i2 in df_new_severe2:
    numbers_c2=numbers_c2-1
numbers_c2=int(numbers_c2/people_c_severe2)
for i1 in df_new_severe1:
    numbers_c1=numbers_c1-1
numbers_c1=int(numbers_c1/people_c_severe1)

#Запись ковидных в списки по тяжести заболевания
spectr_covid3=np.array(df_new_severe3['Spectrum_c_severe3'])
spectr_covid2=np.array(df_new_severe2['Spectrum_c_severe2'])
spectr_covid1=np.array(df_new_severe1['Spectrum_c_severe1'])

#Чтение здоровых
wavelength_h = []
for file in f_healthy:
    read_healthy = pd.read_csv(dir_healthy + file, sep='\t')
    wavel = np.asarray(read_healthy['Wl'].astype(float))
    spectr_h = np.asarray(read_healthy['OAD'].astype(float))
    wavelength_h.extend(wavel)
    # col4h: ndarray = np.array(read_file['OAD'].astype(float))
    people_h = int(people_h + 1)
    spectr_healthy.extend(spectr_h)
    for data in spectr_h:
        numbers_h = numbers_h + 1

#Чистка здоровых
df_h = pd.DataFrame(data=[spectr_healthy, wavelength_h]).T
df_h = df_h.set_axis(['Spectrum_h', 'Wavel'], axis=1, inplace=False)
df_del_h = df_h.loc[(df_h['Wavel'] >= 9.8) & (df_h['Wavel'] <= 10.1)]
delete_h=[]
for wavel_del_h in wavelength_h:
    if (wavel_del_h >= 9.8) & (wavel_del_h <= 10.1):
        delete_h.append(wavel_del_h)
df_new_h = df_h[~df_h['Wavel'].isin(delete)]
for num in df_new_h:
    numbers_h = numbers_h - 1
numbers_h = int(numbers_h / people_h)

#Запись здоровых в список
spectr_healthy = np.array(df_new_h['Spectrum_h'])
#Изменение размерности массивов
data_covid3=np.reshape(spectr_covid3,(people_c_severe3,numbers_c3))
data_covid2=np.reshape(spectr_covid2,(people_c_severe2,numbers_c2))
data_covid1=np.reshape(spectr_covid1,(people_c_severe1,numbers_c1))
data_healthy = np.reshape(spectr_healthy, (people_h, numbers_h))

#Совмещение массивов
together = np.vstack((data_covid1,data_covid2,data_covid3,data_healthy))

#Список статусов
status = []
for i in range(people_c_severe1):
    status.append('Легкая ст')
for i in range(people_c_severe2):
    status.append('Средняя')
for i in range(people_c_severe3):
    status.append('Тяжелая')
for i in range(people_h):
    status.append('Healthy')

axis_font = {'fontname': 'Arial'}
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (16, 9)
col = ['Red', 'Green', 'Blue', 'Black', 'Yellow']

##PCA
pipeline = make_pipeline(StandardScaler(), PCA(n_components=4))
XPCAreduced = pipeline.fit_transform(together)
plt.plot(np.cumsum((pipeline.named_steps.pca.explained_variance_ratio_)))
plt.tight_layout()
plt.savefig('explained_covid.png', dpi=600)
plt.close()
loadings = np.zeros(pipeline.named_steps.pca.components_.shape)

loadings_normed = (pipeline.named_steps.pca.components_.T *
                   np.sqrt(pipeline.named_steps.pca.explained_variance_)).T
loadings = pipeline.named_steps.pca.components_
ind = np.asarray(wavelength)

items_to_skip = 0
items_count = 0
splitted = []
markers = ['o', 'v', 's', '*', '2', 'p', 'D']
for i in range(0, 4):
    if i == 0:
        items_count = people_c_severe1
    elif i == 1:
        items_count = people_c_severe2
        items_to_skip = people_c_severe1
    elif i==2:
        items_count=people_c_severe3
        items_to_skip=people_c_severe2+people_c_severe1
    elif i==3:
        items_count=people_h
        items_to_skip=people_c_severe3+people_c_severe2+people_c_severe1
    splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count],
                     col[i], status[items_to_skip], markers[i]))
#Состоит из 1.Массив ковид 1, ковид2,ковид3, здоровые и прилегающих им элементов для графика
#splitted[0][0]-это первый элемент(массив по ковид 1)
# [0][1]-ковид2
#[0][2]-ковид3
#[0][3]-здоровые
#print('splitted=',splitted[0])
# Визуализация в файлы
n_components = 4
vis_folder_name = 'covid_visualization2d_' + str(n_components)
#print(pipeline.named_steps.pca.explained_variance_ratio_)
if not os.path.exists(vis_folder_name):
    os.mkdir(vis_folder_name)
coordsi=[]
coordsj=[]
count_plots=0
count_figs=0
numb_coord=[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
for i in range(0, n_components):
    for j in range(i + 1, n_components):
        count_group = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        count_figs+=1
        for group in splitted:
            ax.scatter(group[0][:, i], group[0][:, j], alpha=0.8,
                       c=group[1], label=group[2], marker=group[3])
            coordsi.append(group[0][:,i])
            coordsj.append(group[0][:,j])
            #print('groups=',group[0][:, i], group[0][:, j])
            count_people = 0
            if count_figs<=6:
                if count_group <= 2:
                    #print(group[0][count_people][0],group[0][count_people][1])
                    for el in patient[count_group]:
                        ax.annotate(el, xy=(coordsi[numb_coord[count_plots]][count_people],
                                            coordsj[numb_coord[count_plots]][count_people]), size=10)
                        #print('data=',coordsi[count_group*count_figs][count_people],
                                            #coordsj[count_group*count_figs][count_people])
                        count_people += 1
                    count_group += 1
                    count_plots+=1
        ax.legend()
        ax.grid(False)
        plt.xlabel('PC ' + str(i + 1) + ', explained variance ' + str(
            round(100 * pipeline.named_steps.pca.explained_variance_ratio_[i], 1)) + '%')
        plt.ylabel('PC ' + str(j + 1) + ', explained variance ' + str(
            round(100 * pipeline.named_steps.pca.explained_variance_ratio_[j], 1)) + '%')
        plt.tight_layout()
        plt.savefig(vis_folder_name + '\\PCA_i_' + str(i) + '_j_' + str(j)  + '.png',
                    dpi=600)
        plt.close()
#подсчет расстояния:
distance_1=[]
mean_dist=[]
mean_dist2=[]
sum_mean_dist=[]
for i in range(len(splitted[0])-1):   
    for j in range(len(splitted[i][0])):
        for sample in splitted[3][0]:
            cov_mat = np.linalg.pinv(np.cov(np.vstack([splitted[i][0][j],sample]).T))
            distance_xy=distance.mahalanobis(splitted[i][0][j],sample,cov_mat)
            #distance_xy=distance.euclidean(splitted[i][0][j],sample)
            distance_1.append(distance_xy)
        mean_dist.append(np.sum(distance_1)/float(len(splitted[i][0])))
        distance_1=[]
    mean_dist2.append(mean_dist)
    mean_dist=[]
        #sum_mean_dist+=mean_dist2
        #sum_mean_dist=sum_mean_dist[i]/float(numbers_c1)
print(mean_dist2)

for i in range(0, n_components):
    for j in range(0, n_components):
        distance_xy=(math.sqrt(coordsi[0][i]-coordsi[3][j])**2+(coordsj[0][i]-coordsj[3][j])**2)
        distance_1.append(distance_xy)
    mean_dist+=distance_1
    mean_dist2.append(mean_dist[i]/float(numbers_h))
    sum_mean_dist+=mean_dist2
    sum_mean_dist=sum_mean_dist[i]/float(numbers_c1)
print(mean_dist2)
        