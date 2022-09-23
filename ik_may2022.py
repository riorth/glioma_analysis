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
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn import svm
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from scipy import interpolate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import pairwise
from sklearn import cluster
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_selection import RFECV
import seaborn as sns
from io import StringIO
from sklearn.manifold import MDS
import lightgbm as lgb



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

def simple_importance(spectrum_a, spectrum_b):
    #return np.abs(spectrum_a-spectrum_b)/np.maximum(spectrum_a, spectrum_b)
    return np.abs(spectrum_a-spectrum_b)


def main():
    axis_font = {'fontname':'Arial'}
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.figsize"] = (16,9)
    col = ['Red', 'Green', 'Blue', 'Yellow', 'Black']
    if 1: 
    #############################################################################################################
        #msu data analysis
    #############################################################################################################     
        # path_to_U87_1 = os.path.join('data_ik/U87_1/')
        # path_to_U87_2 = os.path.join('data_ik/U87_2/')
        # path_to_U87_3 = os.path.join('data_ik/U87_3/')
        # path_to_U87_4 = os.path.join('data_ik/U87_4/')
        # path_to_control_1 = os.path.join('data_ik/Control_1/')
        # path_to_control_2 = os.path.join('data_ik/Control_2/')
        # path_to_control_3 = os.path.join('data_ik/Control_3/')
        # path_to_control_4 = os.path.join('data_ik/Control_4/')

        if 1: 
            # path_to_U87_1 = os.path.join('data_ik1/U87_1/')
            # path_to_U87_2 = os.path.join('data_ik1/U87_2/')
            # path_to_U87_3 = os.path.join('data_ik1/U87_3/')
            # path_to_U87_4 = os.path.join('data_ik1/U87_4/')
            # path_to_control_1 = os.path.join('data_ik1/Control_1/')
            # path_to_control_2 = os.path.join('data_ik1/Control_2/')
            # path_to_control_3 = os.path.join('data_ik1/Control_3/')
            # path_to_control_4 = os.path.join('data_ik1/Control_4/')
            path_to_U87_1 = os.path.join('data_ik_may2022/U87_1/')
            path_to_control_1 = os.path.join('data_ik_may2022/Control/')

        if 0: 
            path_to_U87_1 = os.path.join('data_ik_absorption/U87_1/')
            path_to_U87_2 = os.path.join('data_ik_absorption/U87_2/')
            path_to_U87_3 = os.path.join('data_ik_absorption/U87_3/')
            path_to_U87_4 = os.path.join('data_ik_absorption/U87_4/')
            path_to_control_1 = os.path.join('data_ik_absorption/Control_1/')
            path_to_control_2 = os.path.join('data_ik_absorption/Control_2/')
            path_to_control_3 = os.path.join('data_ik_absorption/Control_3/')
            path_to_control_4 = os.path.join('data_ik_absorption/Control_4/')        


        frequencies = []
        U87_1 = []

        control_1 = []

    
        # with open(frequency_path_msu) as f:
        #     lines = f.readlines()
        # for line in lines:
        #     line.replace('\n','')
        #     frequencies_msu.append(float(line))
    #############################################################################################################    
        # Чтение данных из файлов
    #############################################################################################################    
        # U87_1
        file_list = os.listdir(path_to_U87_1)
        for fn in file_list:
            with open(os.path.join(path_to_U87_1, fn), encoding='utf-8-sig') as ref:
                result = np.array(list(csv.reader(ref, delimiter=";"))).astype("float")
                #result = np.array(list(csv.reader(ref, delimiter='\t'))).astype("float")
                frequencies.append(result[:,0])
                U87_1.append(result[:,1])

        # Control_1
        file_list = os.listdir(path_to_control_1)
        for fn in file_list:
            with open(os.path.join(path_to_control_1, fn)) as ref:
                result = np.array(list(csv.reader(ref, delimiter=";"))).astype("float")
                control_1.append(result[:,1])       
        
 
        names = frequencies
        print ('done files reading')   
########################################################
# Outlier detection#####################################
# ######################################################        
        if 1:
            iso = IsolationForest(contamination='auto')
            #iso = IsolationForest(contamination=0.1, bootstrap=True)
            yhat = iso.fit_predict(U87_1)
            mask = yhat != -1
            print ('Number of outliers in u1', len(U87_1) - np.sum(mask), 'of total ', len(U87_1))  
            for i in range(0,len(U87_1)):
                if yhat[i]==-1:                    
                    print("Outliers are:",i)
            U87_1 = np.asarray(U87_1)[mask]


            yhat = iso.fit_predict(control_1)
            mask = yhat != -1
            print ('Number of outliers in c3', len(control_1) - np.sum(mask), 'of total ', len(control_1))  
            for i in range(0,len(control_1)):
                if yhat[i]==-1:                    
                    print("Outliers are:",i)            
            control_1 = np.asarray(control_1)[mask]

    #############################################################################################################
        # Формируем массив данных
    #############################################################################################################    
        if 1:
            labels = []
            y = []
            #X = np.zeros((len(U87_1) + len(U87_2) + len(U87_3) + len(U87_4) + len(control_1) + len(control_2) + len(control_3) + len(control_4),len(control_1[0])))
            X = np.zeros((len(U87_1) + len(control_1) ,len(U87_1[0])))
            #X = np.zeros((len(U87_1)+len(U87_3),len(U87_3[0])))
            #X = np.zeros((len(U87_1)+len(U87_2),len(U87_2[0])))
            #X = np.zeros((len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4),len(U87_3[0])))
            i = 0
            j = 0
            k=0
            if 1:
                for sample in U87_1:
                    X[i, 0:len(U87_1[0])] = sample
                    i+=1
                    labels.append('Tumor 1 week')
                    y.append(k)
                k+=1

            if 0:
                for sample in U87_2:
                    X[i, 0:len(U87_2[0])] = sample
                    i+=1
                    labels.append('Tumor 2 week')
                    y.append(k)
                k+=1

            if 0:                
                for sample in U87_3:
                    X[i, 0:len(U87_3[0])] = sample
                    i+=1
                    labels.append('Tumor 3 week')
                    y.append(k)
                k+=1

            if 0:
                for sample in U87_4:
                    X[i, 0:len(U87_4[0])] = sample
                    i+=1
                    labels.append('Tumor 4 week')
                    y.append(k)
                k+=1

            if 1:
                for sample in control_1:
                    X[i, 0:len(control_1[0])] = sample
                    i+=1
                    labels.append('Control 1 week')
                    y.append(k)
                k+=1

            if 0:
                for sample in control_2:
                    X[i, 0:len(control_2[0])] = sample
                    i+=1
                    labels.append('Control 2 week')
                    y.append(k)  
                k+=1

            if 0:      
                for sample in control_3:
                    X[i, 0:len(control_3[0])] = sample
                    i+=1
                    labels.append('Control 3 week')
                    y.append(k)
                k+=1

            if 0:
                for sample in control_4:
                    X[i, 0:len(control_4[0])] = sample
                    i+=1
                    labels.append('Control 4 week')
                    y.append(k)     
            
            

            # X_new = np.copy(X)
            # X=np.copy(np.fliplr(X_new))

            if 0:
                bir = np.array([[1880,2100],[3700,4100]]) 
                #bir = np.array([[500,900],[2100,2700]])                 
                X_new = np.zeros_like(X)
                for i in range(len(X)):
                    #corrected_values, baseline_values = rp.baseline(frequencies[0], X[i,:],bir,'poly',polynomial_order=2 )
                    #corrected_values, baseline_values = rp.baseline(frequencies[0], X[i,:], roi,'unispline',s=1e0)
                    corrected_values, baseline_values = rp.baseline(frequencies[0], X[i,:], bir,'als',lam=10**7,p=0.05)
                    #corrected_values, baseline_values = rp.baseline(frequencies[0], X[i,:], bir,'arPLS',lam=10**5,ratio=0.001)
                    #X_new[i] = X[i,:] - peakutils.baseline(X[i,:])
                    X_new[i] = corrected_values.flatten()
                    
                plt.plot(np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87_1")  
                X=np.copy(X_new)
                plt.plot(np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87_1")  
                plt.show()
                plt.close()

        # u87_2len = len(U87_1)+len(U87_2)
        # u87_3len = len(U87_1)+len(U87_2) + len(U87_3)
        # u87_4len = len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4)        
        # ctrl_2len = len(control_1)+len(control_2)
        # ctrl_3len = len(control_1)+len(control_2)+len(control_3)        
        # ctrl_4len = len(control_1)+len(control_2)+len(control_3) + len(control_4)        
############## Mean and median spectra #############################
        if 0:        
                        
            plt.plot(frequencies[0], np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87 week 1")          
            # plt.plot(frequencies[0], np.mean(X[len(U87_1):u87_2len], axis=0),color='blue', label="mean U87 week 2")
            # plt.plot(frequencies[0], np.mean(X[u87_2len:u87_3len], axis=0),color='red', label="mean U87 week 3")
            # plt.plot(frequencies[0], np.mean(X[u87_3len:], axis=0),color='black', label="mean U87 week 4")



            #plt.plot(frequencies[0], np.mean(X[:len(U87_3)], axis=0),color='red', label="mean Tumor")
            #plt.plot(frequencies[0], np.mean(X[len(U87_3):], axis=0),color='blue', label="mean Control")
            #plt.plot(np.mean(X[u87_3len:u87_4len], axis=0),color='magenta', label="mean U87_4")
            tmp_ticks = np.arange(450, 4000, 200) 
            plt.xticks(tmp_ticks, tmp_ticks[::-1]) 
            plt.legend()
            plt.xlabel('IR frequencies', **axis_font)
            plt.ylabel('IR intensities, Arb. Units', **axis_font)    
            plt.tight_layout()           
            #plt.show()
            plt.savefig('ir_fig_1_mean_spectra.png', dpi = 600)
            plt.close()

            #plt.plot(np.median(X[:len(U87_1)], axis=0),color='green', label="median U87_1")          
            #plt.plot(np.median(X[len(U87_1):u87_2len], axis=0),color='blue', label="median U87_2")
            #plt.plot(frequencies[0], np.median(X[:len(U87_2)], axis=0),color='red', label="median Tumor")
            #plt.plot(frequencies[0], np.median(X[len(U87_2):], axis=0),color='blue', label="median Control")
            #plt.plot(np.median(X[u87_3len:u87_4len], axis=0),color='magenta', label="median U87_4")

            # plt.plot(np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87_1")          
            # plt.plot(np.mean(X[u87_len:u87_len+len(control_1)], axis=0),color='red', label="mean Control_1")          
            # plt.plot(np.mean(X[len(U87_1):len(U87_1)+len(U87_2)], axis=0),color='blue', label="mean U87_2")
            # plt.plot(np.mean(X[len(U87_1)+len(U87_2):len(U87_1)+len(U87_2) + len(U87_3)], axis=0),color='red', label="mean U87_3")
            # plt.plot(np.mean(X[len(U87_1)+len(U87_2)+len(U87_3):len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4)], axis=0),color='magenta', label="mean U87_4")

            # plt.plot(np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87_1")          
            # plt.plot(np.mean(X[len(U87_1):len(U87_1)+len(U87_2)], axis=0),color='blue', label="mean U87_2")
            #plt.plot(np.mean(X[len(U87_1)+len(U87_2):len(U87_1)+len(U87_2) + len(U87_3)], axis=0),color='red', label="mean U87_3")
            #plt.plot(np.median(X[len(U87_1)+len(U87_2):len(U87_1)+len(U87_2) + len(U87_3)], axis=0),color='blue', label="median U87_3")
            #plt.plot(np.mean(X[len(U87_1)+len(U87_2)+len(U87_3):len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4)], axis=0),color='red', label="mean U87_4")                        
            #plt.plot(np.median(X[len(U87_1)+len(U87_2)+len(U87_3):len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4)], axis=0),color='blue', label="median U87_4")                        

            plt.plot(frequencies[0], np.median(X[:len(U87_1)], axis=0),color='green', label="median U87 week 1")          
            # plt.plot(frequencies[0], np.median(X[len(U87_1):u87_2len], axis=0),color='blue', label="median U87 week 2")
            # plt.plot(frequencies[0], np.median(X[u87_2len:u87_3len], axis=0),color='red', label="median U87 week 3")
            # plt.plot(frequencies[0], np.median(X[u87_3len:], axis=0),color='black', label="median U87 week 4")            
            tmp_ticks = np.arange(450, 4000, 200) 
            plt.xticks(tmp_ticks, tmp_ticks[::-1]) 
            plt.legend()
            plt.xlabel('IR frequencies', **axis_font)
            plt.ylabel('IR intensities, Arb. Units', **axis_font)               
            plt.tight_layout()
            #plt.show()
            plt.savefig('IR_fig_1_median_spectra.png', dpi = 600)
            plt.close()


###################################################################            
        #pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
        pipeline=make_pipeline(PCA(n_components=7))
        XPCAreduced=pipeline.fit_transform(X) 
        plt.plot(np.cumsum((pipeline.named_steps.pca.explained_variance_ratio_)))
        plt.tight_layout()
        plt.savefig('fig_3.png', dpi = 600)
        plt.close()     
        #pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
        #XPCAreduced=pipeline.fit_transform(X)             
        #plt.show()  
        n_components = 4
        loadings = np.zeros(pipeline.named_steps.pca.components_.shape)
        for i in range(len(pipeline.named_steps.pca.components_)):
            loadings[i,:] = pipeline.named_steps.pca.components_[i,:] * np.sqrt(pipeline.named_steps.pca.explained_variance_)[i]
        ind = np.asarray(frequencies[0])


        ###### Plot errorbar###############
        if 0:
            #### 3 week #######
            plt.errorbar(frequencies[0], np.mean(X[:len(U87_3)],axis=0), yerr=np.std(X[:len(U87_3)],axis=0),label="U87_3",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[len(U87_3):len(control_3)+len(U87_3)],axis=0), yerr=np.std(X[len(U87_3):len(U87_3)+len(control_3)],axis=0),label="Control_3",alpha = 0.2)
            #plt.plot(frequencies[0], np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), label="Control_1", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()
        if 0:    
            #### 1 week #######
            plt.errorbar(frequencies[0], np.mean(X[:len(U87_1)],axis=0), yerr=np.std(X[:len(U87_1)],axis=0),label="U87_1",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[len(U87_1):len(U87_1)+len(U87_2)],axis=0), yerr=np.std(X[len(U87_1):len(U87_1)+len(U87_2)],axis=0),label="U87_2",alpha = 0.2)
            #plt.plot(frequencies[0], np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), label="Control_1", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()
        if 0:
            #### 2 week #######
            plt.errorbar(frequencies[0], np.mean(X[len(U87_1):len(U87_1)+len(U87_2)],axis=0), yerr=np.std(X[len(U87_1):len(U87_1)+len(U87_2)],axis=0),label="U87_2",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_2len:u87_3len],axis=0), yerr=np.std(X[u87_2len:u87_3len],axis=0),label="U87_3",alpha = 0.2)
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()
        if 0:
            #### 3 week ######
            plt.errorbar(frequencies[0], np.mean(X[:len(U87_2)],axis=0), yerr=np.std(X[:len(U87_2)],axis=0),label="U87_2",alpha = 0.8)
            #plt.errorbar(frequencies[0], np.mean(X[:len(U87_2)],axis=0), yerr=np.std(X[:len(U87_2)],axis=0),label="U87_3",alpha = 0.8)
            plt.errorbar(frequencies[0], np.mean(X[len(U87_2):],axis=0), yerr=np.std(X[len(U87_2):],axis=0),label="U87_3",alpha = 0.2)
            plt.plot(frequencies[0], np.mean(X[len(U87_2):],axis=0), label="U87_3", color="red")
            plt.legend()
            plt.xlabel('IR frequencies', **axis_font)
            plt.ylabel('IR intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.savefig('IR_fig_error_bar.png', dpi = 600)
            plt.close()
        if 0:
            #### 4 week #######
            plt.errorbar(frequencies[0], np.mean(X[u87_3len:u87_4len],axis=0), yerr=np.std(X[u87_3len:u87_4len],axis=0),label="U87_4",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0), yerr=np.std(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0),label="Control_4",alpha = 0.1)
            plt.plot(frequencies[0], np.mean(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0), label="Control_4", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

        if 0:
            #### 1 week #######
            plt.errorbar(frequencies[0], np.mean(X[:len(U87_1)],axis=0), yerr=np.std(X[:len(U87_1)],axis=0),label="U87_1",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), yerr=np.std(X[u87_4len:u87_4len+len(control_1)],axis=0),label="Control_1",alpha = 0.1)
            plt.plot(frequencies[0], np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), label="Control_1", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

            #### 2 week #######
            plt.errorbar(frequencies[0], np.mean(X[len(U87_1):u87_2len],axis=0), yerr=np.std(X[len(U87_1):u87_2len],axis=0),label="U87_2",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_4len+len(control_1):u87_4len+ctrl_2len],axis=0), yerr=np.std(X[u87_4len+len(control_1):u87_4len+ctrl_2len],axis=0),label="Control_2",alpha = 0.1)
            plt.plot(frequencies[0], np.mean(X[u87_4len+len(control_1):u87_4len+ctrl_2len],axis=0), label="Control_2", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

            #### 3 week ######
            plt.errorbar(frequencies[0], np.mean(X[u87_2len:u87_3len],axis=0), yerr=np.std(X[u87_2len:u87_3len],axis=0),label="U87_3",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_4len+ctrl_2len:u87_4len+ctrl_3len],axis=0), yerr=np.std(X[u87_4len+ctrl_2len:u87_4len+ctrl_3len],axis=0),label="Control_3",alpha = 0.1)
            plt.plot(frequencies[0], np.mean(X[u87_4len+ctrl_2len:u87_4len+ctrl_3len],axis=0), label="Control_3", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

            #### 4 week #######
            plt.errorbar(frequencies[0], np.mean(X[u87_3len:u87_4len],axis=0), yerr=np.std(X[u87_3len:u87_4len],axis=0),label="U87_4",alpha = 0.9)
            plt.errorbar(frequencies[0], np.mean(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0), yerr=np.std(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0),label="Control_4",alpha = 0.1)
            plt.plot(frequencies[0], np.mean(X[u87_4len+ctrl_3len:u87_4len+ctrl_4len],axis=0), label="Control_4", color="red")
            plt.legend()
            plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
            plt.ylabel('Raman intensities, Arb. Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

        ##################################

    ##########################################################################################################################
    # Loadings visualization
    ##########################################################################################################################  
        if 1:
            #plt.plot(ind, medfilt(loadings[0,:], 31),"-.",color="blue",label="Loadings for PC1")
    #        plt.plot(ind, medfilt(loadings[1,:], 1),".",color="green",label="Loadings for PC1_1")
        #    plt.plot(ind, median_loadings_h,"--",color="black",label="median_loadings_h")
            #plt.plot(ind, medfilt(loadings[1,:], 31),":",color="purple",label="Loadings for PC2")
        #    plt.plot(ind, median_loadings_il14,"-",color="blue",label="median_loadings_il14")
            plt.plot(ind, medfilt(loadings[2,:], 31),"-",color="red",label="Loadings for PC3")
            #plt.plot(ind, median_loadings_il28,"x",color="green",label="median_loadings_il28")
            plt.plot(ind, medfilt(loadings[3,:], 31),"X",color="yellow",label="Loadings for PC4")
            plt.plot(ind, medfilt(loadings[4,:], 31),"X",color="blue",label="Loadings for PC5")
            plt.plot(ind, medfilt(loadings[5,:], 31),"X",color="green",label="Loadings for PC6")
            #plt.plot(ind, medfilt(loadings[6,:], 31),"X",color="green",label="Loadings for PC7")
            plt.xticks(np.arange(0, 4000, 300)) 
            tmp_ticks = np.arange(450, 4000, 200) 
            #plt.xticks(tmp_ticks, tmp_ticks[::-1])             
    #        plt.yticks(np.arange(0, 11, 1)) 
            plt.xlabel('IR frequencies', **axis_font)
            plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
            plt.legend() 
            plt.tight_layout()
            plt.savefig('fig_loadings.png', dpi = 600)
            #plt.show()
            plt.close() 

            windows_size = 15
            polyorder = 2
            deriv = 0
            plt.plot(ind, savgol_filter(loadings[0,:], windows_size, polyorder, deriv),"-",color="blue",label="Loadings for PC1")
            #plt.plot(ind, savgol_filter(loadings[1,:], windows_size, polyorder, deriv),"-",color="purple",label="Loadings for PC2")
            plt.plot(ind, savgol_filter(loadings[2,:], windows_size, polyorder, deriv),"-",color="purple",label="Loadings for PC3")
            #plt.plot(ind, savgol_filter(loadings[3,:], windows_size, polyorder, deriv),"-",color="purple",label="Loadings for PC4")
            #plt.plot(ind, savgol_filter(loadings[4,:], windows_size, polyorder, deriv),"-",color="red",label="Loadings for PC5")
            #plt.plot(ind, savgol_filter(loadings[5,:], windows_size, polyorder, deriv),"-",color="magenta",label="Loadings for PC6")
            #plt.axhline(y=np.sqrt(1/loadings[0].size), linestyle="--", color="black", label="Cutoff line")
            #plt.axhline(y=-np.sqrt(1/loadings[0].size), linestyle="--", color="black")                      
            #tmp_ticks = np.arange(450, 4000, 200) 
            #plt.xticks(tmp_ticks, tmp_ticks[::-1]) 
            plt.xlabel('IR frequencies', **axis_font)
            plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
            plt.legend() 
            plt.tight_layout()
            plt.savefig('ir_fig_loadings_not_normed_corr.png', dpi = 600)
            #plt.show()
            plt.close()                 

    #        return 0    
    ##########################################################################################################################
    # Explained variance visualization         
    ##########################################################################################################################  
        if 1:        
            features = range(1, pipeline.named_steps.pca.n_components_ + 1)
            plt.bar(features, pipeline.named_steps.pca.explained_variance_ratio_, color='black')
            plt.xlabel('Principal Components')
            plt.ylabel('Explained variance')
            plt.xticks(features)
            l = brocken_stick(pipeline.named_steps.pca.n_components_)
            plt.plot(features, l)
            plt.tight_layout()
            plt.savefig('fig_2.png', dpi = 600)
            plt.close()
            #plt.show()
            # return 1


    # Визуализация
        items_to_skip = 0
        items_count = 0
        splitted = []
        markers = ['o', 'v', 's', '2', '*', 'p', 'D']
        for i in range(0,2):
            if i == 0:
                items_count = len(U87_1)
            elif i == 1:
                #items_count = len(U87_2)
                items_count = len(control_1)
#                items_to_skip += len(U87_2)
            elif i == 2:
                items_count = len(U87_3)
            elif i == 3:
                items_count = len(U87_4)
            elif i == 4:
                items_count = len(control_1)
            elif i == 5:
                items_count = len(control_2)                
            elif i == 6:
                items_count = len(control_3)                           
            else:
                items_count = len(control_4)   

            #splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count], "#%06x" % np.random.randint(0, 0xFFFFFF), labels[items_to_skip], markers[i]))
            splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count], col[i], labels[items_to_skip], markers[i]))
            items_to_skip += items_count

    #MDS
        if 0:
            n_components = 2
            pipeline_scaler=make_pipeline(MDS(n_components=2))
            XPCAreduced=pipeline_scaler.fit_transform(X) 
            cols = ['red', 'blue', 'orange', 'yellow', 'green', 'purple']
            # Визуализация в файлы
            vis_folder_name = 'mds_visualization2d_' + str(n_components)
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
                    #plt.xlabel('PC ' + str(i + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[i], 1)) + '%' )
                    #plt.ylabel('PC ' + str(j + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[j], 1)) + '%' )
                    plt.tight_layout()
                    plt.savefig(vis_folder_name + '\\PCA_i_' + str(i) + '_j_' + str(j) + '.png', dpi = 600)
                    plt.close()  
        # Визуализация в файлы
        if 1:
            n_components = 7
            vis_folder_name = 'ik_visualization2d_' + str(n_components)
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

    #############################################################################################################
    #Finished MSU data analysis
    #############################################################################################################
        if 0:           
            print ('Perform SFFS')
            random_state = 42
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = random_state, stratify=y)
            clf = svm.SVC(kernel='linear')

            # Build step forward feature selection
            if 1:
                sfs_forward = sfs(clf,
                    k_features=(2,10),
                    forward=True,
                    floating=False,
                    verbose=1,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=-1)                    
                sfs_forward = sfs_forward.fit(X_train, y_train)
            # Which features?
                feat_cols = list(sfs_forward.k_feature_idx_)
                print('best combination (ACC: %.3f): %s\n' % (sfs_forward.k_score_, sfs_forward.k_feature_idx_))
                print('all subsets:\n', sfs_forward.subsets_)
                plot_sfs(sfs_forward.get_metric_dict(), kind='std_err')      
                print(feat_cols)
                print(sfs_forward.k_feature_names_)

        # Build step backward feature selection
            if 1:
                sfs_backward = sfs(clf,
                    k_features=(2,10),
                    forward=False,
                    floating=False,
                    verbose=0,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=-1)
            
        # Perform SFFS
                sfs_backward = sfs_backward.fit(X_train, y_train)
        # Which features?
                feat_cols = list(sfs_backward.k_feature_idx_)
                print(feat_cols)
                print(sfs_backward.k_feature_names_)

                clf.fit(X_train, y_train)
                print('classifier accuracy:', clf.score(X_test,y_test))


            if 0:
                rfe = RFECV(clf, step=1, cv=3, n_jobs=-1)
                fit = rfe.fit(X_train, y_train)
                print("RFECV linear svm")
                print("Optimal number of features : %d" % rfe.n_features_)

                # Plot number of features VS. cross-validation scores
                plt.figure()
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (nb of correct classifications)")
                plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
                plt.tight_layout()
                plt.savefig('cv_scores_svm.png', dpi = 600)
                plt.close()                                                                      
                print("Num Features: %s" % (fit.n_features_))
                print("Selected Features: %s" % (fit.support_))
                print("Feature Ranking: %s" % (fit.ranking_).tolist())

                clf = RandomForestClassifier(n_estimators=100, bootstrap = True)
                rfe = RFECV(clf, step=1, cv=3, n_jobs=-1)
                fit = rfe.fit(X, y)
                print("RFECV random forest")    
                plt.figure()
                plt.xlabel("Number of features selected")
                plt.ylabel("Cross validation score (nb of correct classifications)")
                plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
                plt.tight_layout()
                plt.savefig('cv_scores_rf.png', dpi = 600)
                plt.close()                                                               
                print("Num Features: %s" % (fit.n_features_))
                print("Selected Features: %s" % (fit.support_))
                print("Feature Ranking: %s" % (fit.ranking_).tolist())       



    # shrinked=X[:,lbound:ubound]
    # plt.plot(frequencies_shrinked,shrinked.transpose())  
    # plt.xlabel('Frequency, THz', **axis_font)
    # plt.ylabel('Intensity, a.u.', **axis_font)   
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('fig_0.png', dpi = 600)
    # #plt.show()
    # plt.close()
    # plt.plot(frequencies,X.transpose())  
    # plt.xlabel('Frequency, THz', **axis_font)
    # plt.ylabel('Intensity, a.u.', **axis_font)   
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('fig_0_full.png', dpi = 600)
    # #plt.show()
    # plt.close()

    # pipeline=make_pipeline(StandardScaler(), PCA(n_components=10))
    # XPCAreduced=pipeline.fit_transform(shrinked)
    # n_components = 2
    # np.savetxt('pca_var.txt', pipeline.named_steps.pca.explained_variance_ratio_)
    # pca = PCA(n_components=n_components)
    # XPCAreduced = pca.fit_transform(X)

# #############################################################################
# Classification and ROC analysis for Linear SVM
# Run classifier with cross-validation and plot ROC curves
    if 1:
        num_features = len(frequencies[0])
        features = frequencies[0]
        #bound = len(U87_1) + len(control_1)
        #X_sub = XPCAreduced[:bound]
        #y_sub = np.asarray(y[:bound])
        #pipeline_scaler=make_pipeline(StandardScaler())
        #X_sub = pipeline_scaler.fit_transform(X)
        n_components_svm = 10
        pipeline=make_pipeline(StandardScaler(), PCA(n_components=n_components_svm))
        XPCAreduced=pipeline.fit_transform(X)  
        X_sub = np.copy(XPCAreduced)
        y_sub = np.copy(y)

        print("#############Linear SVM#############")
        random_state = np.random.RandomState(14)
        #n_splits=3
        #cv = StratifiedKFold(n_splits=n_splits)
        n_splits=10
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3)        
        #classifier = svm.SVC(kernel='linear', class_weight="balanced", probability=True,
        #                    random_state=random_state)
        classifier = svm.SVC(kernel='linear', class_weight="balanced", probability=True)        

        tprs = []
        aucs = []
        precs = []
        recs = []
        confusion_matrices = []
        c_reps = []
        classification_reports = []
        importances_ave = []
        mean_fpr = np.linspace(0, 1, 100)       
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(X_sub, y_sub)):
            classifier.fit(X_sub[train], y_sub[train])
            viz = plot_roc_curve(classifier, X_sub[test], y_sub[test],
                                name='ROC fold {}'.format(i),
    #                            name='_nolegend_',
                                alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            precs.append(precision_score(y_sub[test], classifier.predict(X_sub[test]),  average='weighted'))
            recs.append(recall_score(y_sub[test], classifier.predict(X_sub[test]),  average='weighted'))
            confusion_matrices.append(confusion_matrix(y_sub[test], classifier.predict(X_sub[test]), labels=[0, 1]))
            c_reps.append(classification_report(y_sub[test], classifier.predict(X_sub[test]), target_names=["Healthy", "Tumor"]))
            importances_ave.append(abs(classifier.coef_[0]))

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)          
        print("Average Recall", np.mean(recs, axis=0))
        print("Average Precision", np.mean(precs, axis=0))
        c_sum=np.zeros_like(confusion_matrices[0])
        for i in range (n_splits): 
            c_sum=c_sum+confusion_matrices[i]   
        print(np.mean(confusion_matrices,axis=0))
        print(np.std(confusion_matrices,axis=0))
        sn, sp = evaluate_sp_sn(c_sum)
        print(f'sensitivity= {sn:.2f}, specificity= {sp:.2f}') 
        acc, prec = evaluate_acc_prec(c_sum)
        print(f'accuracy= {acc:.2f}, precision= {prec:.2f}') 
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        plt.tight_layout()
        #plt.show()
        plt.savefig('IR_fig_SVM_ROC.png', dpi = 600)
        plt.close()

        importances = np.mean(importances_ave, axis=0)
        sum_importances = np.sum(importances)
        importances /= sum_importances
        #indices = (importances).argsort()[::-1]   

        #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
        plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        

        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False)      # ticks along the bottom edge are off       
        #labelbottom=False # labels along the bottom edge are off
        
#        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
        #plt.xticks(range(0,3600,400), range(200,3800,400))
        plt.xticks(range(1, n_components_svm + 1), range(1, n_components_svm + 1))
        #plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)   
        plt.xlabel('Principal сomponent, number', **axis_font)   
        #plt.xlabel('Feature ID')
        plt.ylabel('Relative importance', **axis_font)
        plt.tight_layout()
        #plt.show()    
        plt.savefig('IR_fig_SVM_importance.png', dpi = 600)
        plt.close()
        # features_new= []
        # importances_new = []
        # features_new.append('%.2f'%(features[indices[0]]))
        # importances_new.append(importances[indices[0]])
        # for j in indices[1:num_features]:
        #     for i,feat in enumerate(features_new):
        #         if '%.2f'%(features[j]) == feat:
        #             importances_new[i] += importances[j] 
        #             break
        #     else:
        #         features_new.append('%.2f'%(features[j]))  
        #         importances_new.append(importances[j])  
        # print(features_new)

####### ROC for random forest################
    if 1:
        num_features = len(frequencies[0][200:])
        features = frequencies[0][200:]
        #bound = len(U87_1) + len(control_1)
        #X_sub = XPCAreduced[:bound]
        #y_sub = np.asarray(y[:bound])
        pipeline=make_pipeline(StandardScaler())
        X_sub = pipeline.fit_transform(X[:,200:])        
        #X_sub = np.copy(X[:,200:])
        y_sub = np.copy(y)

        print("#############Random forest#############")
        random_state = np.random.RandomState(42)
        n_splits=3
        cv = StratifiedKFold(n_splits=n_splits)
        classifier = RandomForestClassifier(n_estimators=100, bootstrap = True)

        tprs = []
        aucs = []
        precs = []
        recs = []
        confusion_matrices = []
        c_reps = []
        classification_reports = []
        importances_ave = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(X_sub, y_sub)):
            classifier.fit(X_sub[train], y_sub[train])
            viz = plot_roc_curve(classifier, X_sub[test], y_sub[test],
                                name='ROC fold {}'.format(i),
    #                            name='_nolegend_',
                                alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            precs.append(precision_score(y_sub[test], classifier.predict(X_sub[test]),  average='weighted'))
            recs.append(recall_score(y_sub[test], classifier.predict(X_sub[test]),  average='weighted'))
            confusion_matrices.append(confusion_matrix(y_sub[test], classifier.predict(X_sub[test]), labels=[0, 1]))
            c_reps.append(classification_report(y_sub[test], classifier.predict(X_sub[test]), target_names=["Healthy", "Tumor"]))
            importances_ave.append(classifier.feature_importances_)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)          
        print("Average Recall", np.mean(recs, axis=0))
        print("Average Precision", np.mean(precs, axis=0))
        c_sum=np.zeros_like(confusion_matrices[0])
        for i in range (n_splits): 
            c_sum=c_sum+confusion_matrices[i]   
        print(np.mean(confusion_matrices,axis=0))
        print(np.std(confusion_matrices,axis=0))
        sn, sp = evaluate_sp_sn(c_sum)
        print(f'sensitivity= {sn:.2f}, specificity= {sp:.2f}') 
        acc, prec = evaluate_acc_prec(c_sum)
        print(f'accuracy= {acc:.2f}, precision= {prec:.2f}') 
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        plt.tight_layout()
        #plt.show()
        plt.savefig('IR_fig_RF_ROC.png', dpi = 600)
        plt.close()

        importances = np.mean(importances_ave, axis=0)
        sum_importances = np.sum(importances)
        importances /= sum_importances
        #indices = (importances).argsort()[::-1]   

        #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
        #plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        

        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False)      # ticks along the bottom edge are off       
        #labelbottom=False # labels along the bottom edge are off
        
        plt.xticks(range(0,3600,400), range(200,3800,400))
        plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)   
        #plt.xlabel('Feature ID')
        plt.ylabel('Relative importance', **axis_font)
        plt.tight_layout()
        #plt.show()    
        plt.savefig('IR_fig_RF_importance.png', dpi = 600)
        plt.close()


    # if 0:
    #     plt.figure(figsize=(10,8))
    #     bound = len(U87_1) + len(U87_2)
    #     clf_svm = svm.SVC(kernel='linear', C = 1.0)
    #     clf_svm.fit(XPCAreduced[:bound], y[:bound])
    #     w = clf_svm.coef_[0]
    #     a = -w[0] / w[1]

    #     xx = np.linspace(-7.5,11)
    #     yy = a * xx - clf_svm.intercept_[0] / w[1]

    #     h0 = plt.plot(xx, yy, 'k-')
    #     #h0 = plt.plot(xx, yy, 'k-', label='Decision boundary')
    #     X_set, y_set = XPCAreduced[:bound], y[:bound]
    #     for i, j in enumerate(np.unique(y_set)):
    #         if j == 0:
    #             plt.scatter(X_set[y_set == j+1, 0], X_set[y_set == j+1, 1],
    #                     c = ListedColormap(('red', 'green'))(i), marker='^', label = 'Diabetic')
    #         else:
    #             plt.scatter(X_set[y_set == j-1, 0], X_set[y_set == j-1, 1],
    #                     c = ListedColormap(('red', 'green'))(i), marker='o', label = 'Non-diabetic')
    #     #plt.title('SVM Classifier (Decision boundary for Diabetes vs Control)')
    #     plt.xlabel('Principal Component 1', **axis_font)
    #     plt.ylabel('Principal Component 2', **axis_font)
    #     plt.legend(loc="lower right")
    #     plt.tight_layout()
    #     plt.savefig('fig_svm.png', dpi = 600)
    #     plt.close()
    #     #plt.show()

    #     plt.clf() 
    #     perceptron = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, 
    #                             max_iter=1000, tol=0.0001, shuffle=True, verbose=1, eta0=1.0, 
    #                             n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, 
    #                             n_iter_no_change=50, class_weight=None, warm_start=False)
    #     perceptron.fit(XPCAreduced, y)
    #     print(perceptron.score(XPCAreduced, y))
    #     predicted = perceptron.predict(XPCAreduced)
    #     X_set, y_set = XPCAreduced, y
    #     X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    #                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    #     plt.contourf(X1, X2, perceptron.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #                 alpha = 0.75, cmap = ListedColormap(('navajowhite', 'darkkhaki')))
    #     plt.xlim(X1.min(), X1.max())
    #     plt.ylim(X2.min(), X2.max())
    #     for i, j in enumerate(np.unique(y_set)):
    #         if j == 0:
    #             plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                     c = ListedColormap(('green', 'red'))(i), marker='^', label = 'Non-diabetic')
    #         else:
    #             plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                     c = ListedColormap(('green', 'red'))(i), marker='o', label = 'Diabetic')
    #     #plt.title('Perceptron Classifier (Decision boundary for Diabet vs Control)')
    #     plt.xlabel('Principal Component 1', **axis_font)
    #     plt.ylabel('Principal Component 2', **axis_font)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('fig_perceptron.png', dpi = 600)
    #     plt.close()

    # ###############################################################################################
    # #Working with Random forest classifier
    # ###############################################################################################
    # # pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
    # # XPCAreduced=pipeline.fit_transform(shrinked)
    # # n_components = 4
    # num_features = len(frequencies[0][200:])
    # features = frequencies[0][200:]
    # #bound = len(U87_1) + len(control_1)
    # #X_sub = XPCAreduced[:bound]
    # #y_sub = np.asarray(y[:bound])
    # X_sub = np.copy(X[:,200:])
    # y_sub = np.copy(y)    
    # names=features
    # clf2 = RandomForestClassifier(n_estimators=100, bootstrap = True)

    # clf2.fit(shrinked,y)
    # for name, importance in zip(names, clf2.feature_importances_):
    #     print(name, "=", importance)

    # features = names
    # importances = clf2.feature_importances_
    # indices = (importances).argsort()[::-1]
    # indices_cut = indices[:10]
    # #plt.title('Feature Importances')
    # plt.bar(range(len(indices_cut)), importances[indices_cut], color='b', align='center')
    # plt.xticks(rotation=90)
    # plt.xticks(range(len(indices_cut)), [round(features[i],2) for i in indices_cut])
    # plt.xlabel('Frequency, THz', **axis_font)
    # plt.ylabel('Relative Importance, %', **axis_font)
    # plt.tight_layout()
    # plt.savefig('fig_importance.png', dpi = 600)
    # plt.close()
    # #plt.show()    
    # print(clf2.score(shrinked,y))
    ###############################################################################################
    #Done Working with Random forest classifier
    ###############################################################################################

    return 0

if __name__ == '__main__':
    main()