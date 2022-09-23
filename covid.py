import json
import os
import numpy as np
import sys
import csv
import rampy as rp
import peakutils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import pandas as pd

def brocken_stick(n):
    l=np.zeros(n)
    l[n-1] = 1 / n /n
    for i in range(n-2, -1, -1):
        l[i] =  1/(i+1)/n + l[i+1]
    return l

def evaluate_sp_sn(confusion_matrix):
    conf_mat = np.ravel(confusion_matrix)
    sensitivity = conf_mat[0] / (conf_mat[0] + conf_mat[1])
    specificity = conf_mat[3] / (conf_mat[2] + conf_mat[3])         
    return sensitivity, specificity

def evaluate_acc_prec(confusion_matrix):
    conf_mat = np.ravel(confusion_matrix)
    accuracy = (conf_mat[0] + conf_mat[3])/ (conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3])
    precision = conf_mat[0] / (conf_mat[0] + conf_mat[2])         
    return accuracy, precision


def main():
    axis_font = {'fontname':'Arial'}
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.figsize"] = (16,9)
    col = ['Red', 'Green', 'Blue', 'Yellow', 'Black']
    if 1: 
    #############################################################################################################
        # data analysis
    #############################################################################################################     
        path_to_data = os.path.join('test_covid/')

        frequencies = []
        class_1 = []
        class_2 = []

        control_1 = []
        control_2 = []

        frequency_path_thz = path_to_data + 'freq.txt'
        frequencies = np.genfromtxt(frequency_path_thz)

    #############################################################################################################    
        # Чтение данных из файлов
    #############################################################################################################    
        # class_1
        covid = pd.read_csv(path_to_data+'disease.csv', sep=' ')         
        healthy = pd.read_csv(path_to_data+'healthy.csv', sep=' ')                
        data_covid = np.asarray(covid['data'].astype(float))
        data_healthy = np.asarray(healthy['data'].astype(float))
        data_covid_mat = np.reshape(data_covid, (116, 89))
        data_healthy_mat = np.reshape(data_healthy, (50, 89))
        X = np.vstack((data_covid_mat,data_healthy_mat))
        covid_count = 116
        healthy_count = 50
        labels = []
        for i in range(116):
            labels.append('covid')
        for i in range(50):
            labels.append('healthy')
        print ('done')            
   
###################################################################
        pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
        #pipeline=make_pipeline(PCA(n_components=4))
        XPCAreduced=pipeline.fit_transform(X) 
        plt.plot(np.cumsum((pipeline.named_steps.pca.explained_variance_ratio_)))
        plt.tight_layout()
        plt.savefig('explained_covid.png', dpi = 600)
        plt.close()     
        #pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
        #XPCAreduced=pipeline.fit_transform(X)             
        #plt.show()  
        #n_components = 4
        loadings = np.zeros(pipeline.named_steps.pca.components_.shape)
        # for i in range(len(pipeline.named_steps.pca.components_)):
        #     loadings[i,:] = pipeline.named_steps.pca.components_[i,:] * np.sqrt(pipeline.named_steps.pca.explained_variance_)[i]
        loadings_normed = (pipeline.named_steps.pca.components_.T * np.sqrt(pipeline.named_steps.pca.explained_variance_)).T
        loadings = pipeline.named_steps.pca.components_
        ind = np.asarray(frequencies)


    # Визуализация
        items_to_skip = 0
        items_count = 0
        splitted = []
        markers = ['o', 'v', 's', '2', '*', 'p', 'D']
        for i in range(0,2):
            if i == 0:
                items_count = covid_count
            elif i == 1:
                items_count = healthy_count
                items_to_skip = covid_count

            #splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count], "#%06x" % np.random.randint(0, 0xFFFFFF), labels[items_to_skip], markers[i]))
            splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count], col[i], labels[items_to_skip], markers[i]))
            items_to_skip += items_count

        # Визуализация в файлы
        n_components = 4
        vis_folder_name = 'covid_visualization2d_' + str(n_components)
        if not os.path.exists(vis_folder_name):
            os.mkdir(vis_folder_name)
        for i in range(0, n_components):
            for j in range(i+1, n_components):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for group in splitted:
                    ax.scatter(group[0][:, i], group[0][:, j], alpha=0.8, c = group[1], label = group[2], marker=group[3])
                ax.legend()
                ax.grid(False)
                plt.xlabel('PC ' + str(i + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[i], 1)) + '%' )
                plt.ylabel('PC ' + str(j + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[j], 1)) + '%' )
                plt.tight_layout()
                plt.savefig(vis_folder_name + '\\PCA_i_' + str(i) + '_j_' + str(j) + '.png', dpi = 600)
                plt.close()
        print('done')
 


    return 0

if __name__ == '__main__':
    main()