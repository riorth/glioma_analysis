import json
import os
import numpy as np
import sys
import csv
import rampy as rp
import peakutils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
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
from sklearn.model_selection import StratifiedShuffleSplit #RepeatedKFold

from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from scikitplot.metrics import precision_recall_curve
from scikitplot.metrics import plot_precision_recall_curve
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.signal import argrelextrema
from fnmatch import fnmatch
from catboost import CatBoostClassifier
from scipy.signal import argrelmax
from scipy.signal import find_peaks
import pickle
from contextlib import redirect_stdout
import pySuStaIn
import sklearn.cluster
from sklearn.manifold import TSNE
import pandas
import statsmodels.formula.api as smf
from scipy import stats

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
    if (conf_mat[0] + conf_mat[2]) == 0:
        precision = 0
    else:
        precision = conf_mat[0] / (conf_mat[0] + conf_mat[2])         
    return accuracy, precision

def l1_normalize_mat(X):
    L1_norm = np.linalg.norm(X,1,0)
    return X / L1_norm

def main():
    axis_font = {'fontname':'Arial'}
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.figsize"] = (16,9)
    col = ['Red', 'Green', 'Blue', 'Yellow', 'Black']
    if 1: 
    #############################################################################################################
        #msu data analysis
    #############################################################################################################     
        path_to_U87_1 = os.path.join('data_thz/U87_1/')
        path_to_U87_2 = os.path.join('data_thz/U87_2/')
        path_to_U87_3 = os.path.join('data_thz/U87_3/')
        path_to_U87_4 = os.path.join('data_thz/U87_4/')
        path_to_control_1 = os.path.join('data_thz/Control_1/')
        path_to_control_2 = os.path.join('data_thz/Control_2/')
        path_to_control_3 = os.path.join('data_thz/Control_3/')
        path_to_control_4 = os.path.join('data_thz/Control_4/')             


        frequencies = []
        U87_1 = []
        U87_2 = []
        U87_3 = []
        U87_4 = []
        U87_1_median = []
        U87_2_median = []
        U87_3_median = []
        U87_4_median = []
        control_1 = []
        control_2 = []
        control_3 = []
        control_4 = []
        control_1_median = []
        control_2_median = []
        control_3_median = []
        control_4_median = []

        frequency_path_thz = 'data_thz/freq.txt'
        frequencies = np.genfromtxt(frequency_path_thz, delimiter=',')

    #############################################################################################################    
        # Чтение данных из файлов
    #############################################################################################################    
        cutoff_freq = 234 
        # U87_1
        file_list = os.listdir(path_to_U87_1)
        for fn in file_list:
            with open(os.path.join(path_to_U87_1, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                U87_1_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [U87_1.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # U87_2
        file_list = os.listdir(path_to_U87_2)
        for fn in file_list:
            with open(os.path.join(path_to_U87_2, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                U87_2_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [U87_2.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # U87_3
        file_list = os.listdir(path_to_U87_3)
        for fn in file_list:
            with open(os.path.join(path_to_U87_3, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                U87_3_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [U87_3.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # U87_4
        file_list = os.listdir(path_to_U87_4)
        for fn in file_list:
            with open(os.path.join(path_to_U87_4, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                U87_4_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [U87_4.append(result[:cutoff_freq,i]) for i in range(0,6)]

        # Control_1
        file_list = os.listdir(path_to_control_1)
        for fn in file_list:
            with open(os.path.join(path_to_control_1, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                control_1_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [control_1.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # Control_2
        file_list = os.listdir(path_to_control_2)
        for fn in file_list:
            with open(os.path.join(path_to_control_2, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                control_2_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [control_2.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # Control_3
        file_list = os.listdir(path_to_control_3)
        for fn in file_list:
            with open(os.path.join(path_to_control_3, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                control_2_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [control_3.append(result[:cutoff_freq,i]) for i in range(0,6)]
        # Control_4
        file_list = os.listdir(path_to_control_4)
        for fn in file_list:
            with open(os.path.join(path_to_control_4, fn)) as ref:
                result = np.genfromtxt(ref, delimiter=",")
                control_4_median.append(np.median(result[:cutoff_freq,:],axis = 1))
                [control_4.append(result[:cutoff_freq,i]) for i in range(0,6)]
                       
        frequencies = frequencies[:cutoff_freq] / 1000000000000                                                                        
        

    
        names = frequencies
        print ('done files reading')   

    #############################################################################################################
        # Формируем массив данных
    #############################################################################################################    
        if 1:
            labels = []
            y = []
            group1 = U87_1
            group1_name = "U87_1"
            group2 = U87_2
            group2_name = "U87_2"
            group3 = U87_3
            group3_name = ""
            group4 = U87_4
            group4_name = ""            
            group5 = control_1
            group5_name = ""
            group6 = control_2
            group6_name = ""
            group7 = control_3
            group7_name = ""
            group8 = control_4
            group8_name = ""                        
            #X = np.zeros((len(group1)+len(group2)+len(group3),cutoff_freq))            
            #X = np.zeros((len(group1)+len(group2),cutoff_freq))            
            if 1:
                group1 = U87_1_median
                group2 = U87_2_median
                group3 = U87_3_median
                group5 = control_1_median
                group6 = control_2_median
                group7 = control_3_median
            #X = np.zeros((len(group1)+len(group2)+len(group3)+len(group5)+len(group6)+len(group7),cutoff_freq))            
            #X = np.zeros((len(group1)+len(group2)+len(group3),cutoff_freq))            
            X = np.zeros((len(group1)+len(group2),cutoff_freq))            
            use_pca = 1
            use_normalization = 1
            if use_pca:
                print("PCA is used")
            else: 
                print("PCA is NOT used")
            if use_normalization:
                print("Normalization is used")            
            else: 
                print("Normalization is NOT used")                            
            i = 0
            j = 0
            k=0
            if group1_name == "U87_1":
                for sample in group1:
                    X[i, :] = sample
                    i+=1
                    labels.append('Tumor 1 week')
                    y.append(k)
                k+=1

            if group2_name == "U87_2":
                for sample in group2:
                    X[i, :] = sample
                    i+=1
                    labels.append('Tumor 2 week')
                    y.append(k)
                k+=1

            if group3_name == "U87_3":                
                for sample in group3:
                    X[i, :] = sample
                    i+=1
                    labels.append('Tumor 3 week')
                    y.append(k)
                k+=1

            if group4_name == "U87_4":
                for sample in group4:
                    X[i, :] = sample
                    i+=1
                    labels.append('Tumor 4 week')
                    y.append(k)
                k+=1

            if group5_name == "control_1":
                for sample in group5:
                    X[i, :] = sample
                    i+=1
                    labels.append('Control 1 week')
                    y.append(k)
                k+=1

            if group6_name == "control_2":
                for sample in group6:
                    X[i, :] = sample
                    i+=1
                    labels.append('Control 2 week')
                    y.append(k)  
                k+=1

            if group7_name == "control_3":      
                for sample in group7:
                    X[i, :] = sample
                    i+=1
                    labels.append('Control 3 week')
                    y.append(k)
                k+=1

            if group8_name == "control_4":
                for sample in group8:
                    X[i, :] = sample
                    i+=1
                    labels.append('Control 4 week')
                    y.append(k)     
            
            X_new = np.zeros_like(X)

            #X=np.copy(X[:,:cutoff_freq])

            windows_size = 15
            polyorder = 2
            for i in range(len(X)):
                X_new[i] = savgol_filter(X[i], windows_size, polyorder)
            X=np.copy(X_new)
            if use_normalization:
                X=np.copy(np.divide(X_new,np.reshape(np.max(X_new, axis=1),(X_new.shape[0],1))))

        # week 1
        y_sub_1 = np.array([1.0, 2.9, 3.9, 2.9, 5.1])
        # week 2
        y_sub_2 = np.array([9.7, 10.0, 6.9, 5.8, 15.4, 6.4, 13.5, 11.7, 3.2, 23.0])
        # week 3
        y_sub_3 = np.array([21.0, 139.2, 101.9, 141.8, 75.8, 60.1, 107.9])
        y_sub_all = np.concatenate((y_sub_1, y_sub_2, y_sub_3))
        
        print('done')


############## Cluster analysis #############################
        if 0:

            X_embedded = TSNE(n_components=2, learning_rate='auto', metric = 'cosine',
                            init='random', perplexity=20).fit_transform(X)
            for i,label in enumerate(labels):
                if label =='Tumor 1 week':
                    t1 = plt.scatter(X_embedded[i,0], X_embedded[i,1],c='blue', label = label)
                if label =='Tumor 2 week':
                    t2 = plt.scatter(X_embedded[i,0], X_embedded[i,1],c='red', label = label)
                if label =='Tumor 3 week':
                    t3 = plt.scatter(X_embedded[i,0], X_embedded[i,1],c='green', label = label)
                # if label =='Tumor 4 week':
                #     plt.scatter(X_embedded[i,0], X_embedded[i,1],c='black')
            plt.legend([t1, t2, t3], ['Tumor 1 week', 'Tumor 2 week', 'Tumor 3 week'])
            plt.savefig('fig_thz_tsne.png', dpi = 600)
            plt.close()
        if 1:
            # first a quick look at the patient and control distribution for one of our biomarkers
            
            X_dataframe = pandas.DataFrame(X)
            col_names = ['Freq_' + str(i) for i in range(0,len(names))]
            X_dataframe.columns = col_names
            X_dataframe.insert(loc = 0, column='size', value = pandas.Series(y_sub_all))
            X_dataframe.insert(loc = 0, column='labels', value = pandas.Series(labels))

            biomarkers = X_dataframe.columns[2:10]
            biomarker_test = biomarkers[0]
            sns.displot(data=X_dataframe, # our dataframe
                        x=biomarker_test, # name of the the distribution we want to plot
                        hue='labels', # the "grouping" variable
                        kind='kde') # kind can also be 'hist' or 'ecdf'
            plt.title(biomarker_test)

            plt.close()
            # now we perform the normalization

            # make a copy of our dataframe (we don't want to overwrite our original data)
            zdata = pandas.DataFrame(X_dataframe,copy=True)

            # for each biomarker
            for biomarker in biomarkers:
                mod = smf.ols('%s ~ size'%biomarker,  # fit a model finding the effect of age and headsize on biomarker
                            data=X_dataframe[X_dataframe.labels=='Tumor 1 week'] # fit this model *only* to individuals in the control group
                            ).fit() # fit model    
                #print(mod.summary())
                
                # get the "predicted" values for all subjects based on the control model parameters
                predicted = mod.predict(X_dataframe[['size',biomarker]]) 
                
                # calculate our zscore: observed - predicted / SD of the control group residuals
                w_score = (X_dataframe.loc[:,biomarker] - predicted) / mod.resid.std()
                
                #print(np.mean(w_score[data.Diagnosis==0]))
                #print(np.std(w_score[data.Diagnosis==0]))
                
                # save zscore back into our new (copied) dataframe
                zdata.loc[:,biomarker] = w_score
                
                
            plt.figure(0)
            sns.scatterplot(x=X_dataframe.Freq_0,y=X_dataframe.Freq_1,hue=X_dataframe.labels)
            plt.figure(1)
            sns.scatterplot(x=zdata.Freq_0,y=zdata.Freq_1,hue=zdata.labels)            
            biomarker = biomarkers[0]
            sns.displot(data=zdata,x=biomarker,hue='labels',kind='kde')
            plt.title(biomarker)
            plt.axvline(0,ls='--',c='black') # the 0 line *should* be the mean of the control distribution
            
            plt.close()
            #plt.show()

            N = len(biomarkers)         # number of biomarkers

            SuStaInLabels = biomarkers
            Z_vals = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
            Z_max  = np.array([5]*N)           # maximum z-score
            print(Z_vals)
            # Input the settings for z-score SuStaIn
            # To make the tutorial run faster I've set 
            # N_startpoints = 10 and N_iterations_MCMC = int(1e4)
            # I recommend using N_startpoints = 25 and 
            # N_iterations_MCMC = int(1e5) or int(1e6) in general though

            N_startpoints = 10
            N_S_max = 3
            N_iterations_MCMC = int(1e4)
            output_folder = os.path.join(os.getcwd(), 'PysustainOutput')
            dataset_name = 'THz'

            # Initiate the SuStaIn object
            sustain_input = pySuStaIn.ZscoreSustain(
                                        zdata[biomarkers].values,
                                        Z_vals,
                                        Z_max,
                                        SuStaInLabels,
                                        N_startpoints,
                                        N_S_max, 
                                        N_iterations_MCMC, 
                                        output_folder, 
                                        dataset_name, 
                                        False)
            # make the output directory if it's not already created
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)
            samples_sequence,   \
            samples_f,          \
            ml_subtype,         \
            prob_ml_subtype,    \
            ml_stage,           \
            prob_ml_stage,      \
            prob_subtype_stage  = sustain_input.run_sustain_algorithm()

            # for each subtype model
            for s in range(N_S_max):
                # load pickle file (SuStaIn output) and get the sample log likelihood values
                pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
                pk = pandas.read_pickle(pickle_filename_s)
                samples_likelihood = pk["samples_likelihood"]
                
                # plot the values as a line plot
                plt.figure(0)
                plt.plot(range(N_iterations_MCMC), samples_likelihood, label="subtype" + str(s))
                plt.legend(loc='upper right')
                plt.xlabel('MCMC samples')
                plt.ylabel('Log likelihood')
                plt.title('MCMC trace')
                
                # plot the values as a histogramp plot
                plt.figure(1)
                plt.hist(samples_likelihood, label="subtype" + str(s))
                plt.legend(loc='upper right')
                plt.xlabel('Log likelihood')  
                plt.ylabel('Number of samples')  
                plt.title('Histograms of model likelihood')
                plt.show()
            # Let's plot positional variance diagrams to interpret the subtype progressions

            s = 1 # 1 split = 2 subtypes
            M = len(zdata) 

            # get the sample sequences and f
            pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
            pk = pandas.read_pickle(pickle_filename_s)
            samples_sequence = pk["samples_sequence"]
            samples_f = pk["samples_f"]

            # use this information to plot the positional variance diagrams
            tmp=pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1))

            # let's take a look at all of the things that exist in SuStaIn's output (pickle) file
            pk.keys()
            # The SuStaIn output has everything we need. We'll use it to populate our dataframe.

            s = 1
            pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
            pk = pandas.read_pickle(pickle_filename_s)

            for variable in ['ml_subtype', # the assigned subtype
                            'prob_ml_subtype', # the probability of the assigned subtype
                            'ml_stage', # the assigned stage 
                            'prob_ml_stage',]: # the probability of the assigned stage
                
                # add SuStaIn output to dataframe
                zdata.loc[:,variable] = pk[variable] 

            # let's also add the probability for each subject of being each subtype
            for i in range(s):
                zdata.loc[:,'prob_S%s'%i] = pk['prob_subtype'][:,i]
            zdata.head()
            # IMPORTANT!!! The last thing we need to do is to set all "Stage 0" subtypes to their own subtype
            # We'll set current subtype (0 and 1) to 1 and 0, and we'll call "Stage 0" individuals subtype 0.

            # make current subtypes (0 and 1) 1 and 2 instead
            zdata.loc[:,'ml_subtype'] = zdata.ml_subtype.values + 1

            # convert "Stage 0" subjects to subtype 0
            zdata.loc[zdata.ml_stage==0,'ml_subtype'] = 0
            zdata.ml_subtype.value_counts()

            sns.displot(x='ml_stage',hue='labels',data=zdata,col='ml_subtype')

            sns.pointplot(x='ml_stage',y='prob_ml_subtype', # input variables
              hue='ml_subtype',                 # "grouping" variable
            data=zdata[zdata.ml_subtype>0]) # only plot for Subtypes 1 and 2 (not 0)
            plt.ylim(0,1) 
            plt.axhline(0.5,ls='--',color='k') # plot a line representing change (0.5 in the case of 2 subtypes)


            # Plotting relationship between a biomarker and SuStaIn stage across subtypes

            var = 'Freq_0'

            # plot relationship
            sns.lmplot(x='ml_stage',y=var,hue='ml_subtype',
                    data = zdata[zdata.ml_subtype>0],
                    #lowess=True # uncomment if you would prefer a lowess curve to a linear curve
                    )

            # get stats
            for subtype in [1,2]:
                # get r and p value
                r,p = stats.pearsonr(x = zdata.loc[zdata.ml_subtype==subtype,var].values,
                                    y = zdata.loc[zdata.ml_subtype==subtype,'ml_stage'].values)
                # add them to plot
                plt.text(16,0-subtype,'S%s: r = %s, p = %s'%(subtype,round(r,3),round(p,2)))
            # we can also look at differences in each biomarker across subtypes

            results = pandas.DataFrame(index=biomarkers)
            for biomarker in biomarkers:
                t,p = stats.ttest_ind(zdata.loc[zdata.ml_subtype==0,biomarker],
                                    zdata.loc[zdata.ml_subtype==1,biomarker],)
                results.loc[biomarker,'t'] = t
                results.loc[biomarker,'p'] = p
                
            print(results)
            sns.heatmap(pandas.DataFrame(results['t']),square=True,annot=True,
                    cmap='RdBu_r')
            # plot an example variable:

            var = 'Freq_0'
            sns.boxplot(x='ml_subtype',y=var,data=zdata)
            plt.show()

            # choose the number of folds - here i've used three for speed but i recommend 10 typically
            N_folds = 3

            # generate stratified cross-validation training and test set splits
            labels = zdata.labels.values
            cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
            cv_it = cv.split(zdata, labels)

            # SuStaIn currently accepts ragged arrays, which will raise problems in the future.
            # We'll have to update this in the future, but this will have to do for now
            test_idxs = []
            for train, test in cv_it:
                test_idxs.append(test)
            #test_idxs = np.array(test_idxs,dtype='object')
            test_idxs = np.array(test_idxs,dtype='int')
            # perform cross-validation and output the cross-validation information criterion and
            # log-likelihood on the test set for each subtypes model and fold combination
            CVIC, loglike_matrix     = sustain_input.cross_validate_sustain_model(test_idxs)

            # go through each subtypes model and plot the log-likelihood on the test set and the CVIC
            print("CVIC for each subtype model: " + str(CVIC))
            print("Average test set log-likelihood for each subtype model: " + str(np.mean(loglike_matrix, 0)))

            plt.figure(0)    
            plt.plot(np.arange(N_S_max,dtype=int),CVIC)
            plt.xticks(np.arange(N_S_max,dtype=int))
            plt.ylabel('CVIC')  
            plt.xlabel('Subtypes model') 
            plt.title('CVIC')

            plt.figure(1)
            df_loglike = pandas.DataFrame(data = loglike_matrix, columns = ["s_" + str(i) for i in range(sustain_input.N_S_max)])
            df_loglike.boxplot(grid=False)
            plt.ylabel('Log likelihood')  
            plt.xlabel('Subtypes model') 
            plt.title('Test set log-likelihood across folds')
            plt.show()

            #this part estimates cross-validated positional variance diagrams
            for i in range(N_S_max):
                sustain_input.combine_cross_validated_sequences(i+1, N_folds)
            N_S_selected = 2

            pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1))
            _ = plt.suptitle('SuStaIn output')

            sustain_input.combine_cross_validated_sequences(N_S_selected, N_folds)
            _ = plt.suptitle('Cross-validated SuStaIn output')
            plt.show()

        if 1:
        # Визуализация
            pipeline=make_pipeline(StandardScaler(), PCA(n_components=10))
            #pipeline=make_pipeline(PCA(n_components=4))
            XPCAreduced=pipeline.fit_transform(X) 
            items_to_skip = 0
            items_count = 0
            splitted = []
            markers = ['o', 'v', 's', '2', '*', 'p', 'D']
            for i in range(0,2):
                if i == 0:
                    items_count = len(group1)
                elif i == 1:
                    items_count = len(group3)
                    #items_count = len(control_3)
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

            # Визуализация в файлы
            n_components = 10
            vis_folder_name = 'thz_visualization2d_' + str(n_components)
            if not os.path.exists(vis_folder_name):
                os.mkdir(vis_folder_name)
            for i in range(0, n_components):
                for j in range(i+1, n_components):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    for group in splitted:
                        ax.scatter(group[0][:, i], group[0][:, j], s=256, alpha=0.8, c = group[1], label = group[2], marker=group[3])
                    ax.legend()
                    ax.grid(False)
                    plt.xlabel('PC ' + str(i + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[i], 1)) + '%' )
                    plt.ylabel('PC ' + str(j + 1) + ', explained variance ' + str(round(100*pipeline.named_steps.pca.explained_variance_ratio_[j], 1)) + '%' )
                    plt.tight_layout()
                    plt.savefig(vis_folder_name + '\\PCA_i_' + str(i) + '_j_' + str(j) + '.png', dpi = 600)
                    plt.close()

            for i in range(2,30):
                print('\n' + 'Number of principal components is ' + str(i))
                pipeline=make_pipeline(StandardScaler(), PCA(n_components=i))
                #pipeline=make_pipeline(PCA(n_components=4))
                X_new=pipeline.fit_transform(X) 
                #X = np.copy(XPCAreduced)

                model = cluster.MeanShift()
                labels_pred = model.fit_predict(X_new)
                print('MeanShift adj rand score is ' + str(sklearn.metrics.adjusted_rand_score(labels, labels_pred)))
                model = cluster.DBSCAN(eps=0.30, min_samples=5)
                labels_pred = model.fit_predict(X_new)
                print('DBScan adj rand score is ' + str(sklearn.metrics.adjusted_rand_score(labels, labels_pred)))
                model = cluster.Birch(threshold=0.01, n_clusters=4)
                labels_pred = model.fit_predict(X_new)
                print('BIRCH adj rand score is ' + str(sklearn.metrics.adjusted_rand_score(labels, labels_pred)))
                model = cluster.AgglomerativeClustering(n_clusters=4)
                labels_pred = model.fit_predict(X_new)
                print('AgglomerativeClustering adj rand score is ' + str(sklearn.metrics.adjusted_rand_score(labels, labels_pred)))              
                model = cluster.SpectralClustering(n_clusters=4)
                labels_pred = model.fit_predict(X_new)
                print('SpectralClustering adj rand score is ' + str(sklearn.metrics.adjusted_rand_score(labels, labels_pred)))                                    
            print('Done cluster analysis')






############## Cluster analysis #############################

############## Mean and median spectra #############################
        if 1:        
            ind = np.asarray(frequencies) / 1000000000000
                    
            #1 week
            #plt.plot(ind, np.mean(X[:len(U87_1)], axis=0),color='green', label="mean U87")          
            #plt.plot(ind, np.mean(X[len(U87_1):], axis=0),color='red', label="mean Control")            
            #2 week
            #plt.plot(ind, np.mean(X[:len(U87_2)], axis=0),color='green', label="mean U87")          
            #plt.plot(ind, np.mean(X[len(U87_2):], axis=0),color='red', label="mean Control")        
            #3 week
            plt.plot(ind, np.mean(X[:len(group1)], axis=0),color='red', label="mean Tumor")          
            plt.plot(ind, np.mean(X[len(group1):], axis=0),color='blue', label="mean Control")        

            #plt.plot(ind, np.mean(X[len(U87_1):u87_2len], axis=0),color='blue', label="mean U87_2")
            #plt.plot(ind, np.mean(X[:len(U87_3)], axis=0),color='blue', label="mean U87 3rd week")
            #plt.plot(ind, np.mean(X[u87_3len:], axis=0),color='black', label="mean U87 week 4")
            #x_tmp=normalize(X, norm='l1', axis=1)
            #plt.plot(ind, np.mean(X[:len(control_1)], axis=0),color='green', label="mean Control week 1")          
            # plt.plot(ind, np.mean(X[len(control_1):ctrl_2len], axis=0),color='blue', label="mean Control week 2")

            # plt.plot(ind, np.mean(X[ctrl_3len:], axis=0),color='black', label="mean Control week 4")

            #class_len = len(U87_3)
            #class_len = len(U87_2)
            #class_len = len(U87_1)
            #plt.plot(frequencies, np.mean(X[:class_len], axis=0),color='red', label="mean Tumor")
            #plt.plot(frequencies, np.mean(X[class_len:], axis=0),color='blue', label="mean Control")
            #plt.plot(np.mean(X[u87_3len:u87_4len], axis=0),color='magenta', label="mean U87_4")

            plt.legend()
            plt.xlabel('Frequencies, THz', **axis_font)
            plt.ylabel('Intensity, Arb.Units', **axis_font)    
            plt.tight_layout()           
            #plt.show()
            plt.savefig('fig_1_mean_spectra.png', dpi = 600)
            plt.close()

            # plt.plot(ind, np.median(X[:len(U87_1)], axis=0),color='green', label="median U87 week 1")          
            # plt.plot(ind, np.median(X[len(U87_1):u87_2len], axis=0),color='blue', label="median U87 week 2")
            # plt.plot(ind, np.median(X[u87_2len:u87_3len], axis=0),color='red', label="median U87 week 3")
            # plt.plot(ind, np.median(X[u87_3len:], axis=0),color='black', label="median U87 week 4")

            # plt.plot(ind, np.median(X[:len(control_1)], axis=0),color='green', label="median Control week 1")          
            # plt.plot(ind, np.median(X[len(control_1):ctrl_2len], axis=0),color='blue', label="median Control week 2")
            # plt.plot(ind, np.median(X[ctrl_2len:ctrl_3len], axis=0),color='red', label="median Control week 3")
            # plt.plot(ind, np.median(X[ctrl_3len:], axis=0),color='black', label="median Control week 4")
            #plt.plot(frequencies, np.median(X[:class_len], axis=0),color='red', label="median Tumor")
            #plt.plot(frequencies, np.median(X[class_len:], axis=0),color='blue', label="median Control")
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
            # plt.legend()
            # plt.xlabel('Frequencies, THz', **axis_font)
            # plt.ylabel('Intensity, Arb.Units', **axis_font)               
            # plt.tight_layout()
            # #plt.show()
            # plt.savefig('thz_fig_1_median_spectra.png', dpi = 600)
            # plt.close()
        ###### Plot errorbar###############
            if 1:
                #### 1 week #######
                plt.errorbar(frequencies, np.mean(X[:len(group1)],axis=0), yerr=np.std(X[:len(group1)],axis=0),color='green',label="mean U87",alpha = 0.9)
                plt.errorbar(frequencies, np.mean(X[len(group1):],axis=0), yerr=np.std(X[len(group1):],axis=0),color='red',label="mean Control std",alpha = 0.3)
                plt.plot(frequencies, np.mean(X[len(group1):],axis=0), color="red")
                plt.legend()
                plt.xlabel('Frequencies, THz', **axis_font)
                plt.ylabel('Intensity, Arb.Units', **axis_font)  
                plt.tight_layout()           
                #plt.show()
                plt.savefig('fig_errorbar.png', dpi = 600)
                plt.close()

        ##################################            

###################################################################
        pipeline=make_pipeline(StandardScaler(), PCA(n_components=10))
        #pipeline=make_pipeline(PCA(n_components=4))
        XPCAreduced=pipeline.fit_transform(X) 
        plt.plot(np.cumsum((pipeline.named_steps.pca.explained_variance_ratio_)))
        plt.tight_layout()
        plt.savefig('thz_fig_3.png', dpi = 600)
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


        ###### Plot errorbar###############
        if 0:
            #### 1 week #######
            plt.errorbar(frequencies, np.mean(X[:len(U87_1)],axis=0), yerr=np.std(X[:len(U87_1)],axis=0),label="U87_1",alpha = 0.9)
            plt.errorbar(frequencies, np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), yerr=np.std(X[u87_4len:u87_4len+len(control_1)],axis=0),label="Control_1",alpha = 0.1)
            plt.plot(frequencies, np.mean(X[u87_4len:u87_4len+len(control_1)],axis=0), label="Control_1", color="red")
            plt.legend()
            plt.xlabel('Frequencies, THz', **axis_font)
            plt.ylabel('Intensity, Arb.Units', **axis_font)  
            plt.tight_layout()           
            plt.show()
            plt.close()
        if 0:
            #### 2 week #######
            plt.errorbar(frequencies, np.mean(X[:len(U87_2)],axis=0), yerr=np.std(X[:len(U87_2)],axis=0),label="U87_3",alpha = 0.8)
            plt.errorbar(frequencies, np.mean(X[len(U87_2):],axis=0), yerr=np.std(X[len(U87_2):],axis=0),label="Control_3",alpha = 0.2)
            plt.plot(frequencies, np.mean(X[len(U87_2):],axis=0), label="Control_3", color="red")
            plt.legend()
            plt.xlabel('Frequencies, THz', **axis_font)
            plt.ylabel('Intensity, Arb.Units', **axis_font)  
            plt.tight_layout()    
            plt.savefig('fig_errorbar.png', dpi = 600)       
            #plt.show()
            plt.close()
        if 0:
            #### 3 week ######
            plt.errorbar(frequencies, np.mean(X[:len(U87_3)],axis=0), yerr=np.std(X[:len(U87_3)],axis=0),label="U87 3rd week",alpha = 0.9)
            plt.errorbar(frequencies, np.mean(X[len(U87_3):],axis=0), yerr=np.std(X[len(U87_3):],axis=0),label="Control 3rd week",alpha = 0.3)
            plt.plot(frequencies, np.mean(X[len(U87_3):],axis=0), label="Control 3rd week", color="red")
            plt.legend()
            plt.xlabel('Frequencies, THz', **axis_font)
            plt.ylabel('Intensity, Arb.Units', **axis_font)   
            plt.tight_layout()           
            plt.show()
            plt.close()

        ##################################

    ##########################################################################################################################
    # Loadings visualization
    ##########################################################################################################################  
        if 0:
            plt.plot(ind, medfilt(loadings[0,:], 31),"-.",color="blue",label="Loadings for PC1")
    #        plt.plot(ind, medfilt(loadings[1,:], 1),".",color="green",label="Loadings for PC1_1")
        #    plt.plot(ind, median_loadings_h,"--",color="black",label="median_loadings_h")
            plt.plot(ind, medfilt(loadings[1,:], 31),":",color="purple",label="Loadings for PC2")
        #    plt.plot(ind, median_loadings_il14,"-",color="blue",label="median_loadings_il14")
            plt.plot(ind, medfilt(loadings[2,:], 31),"-",color="red",label="Loadings for PC3")
            #plt.plot(ind, median_loadings_il28,"x",color="green",label="median_loadings_il28")
            plt.plot(ind, medfilt(loadings[3,:], 31),"X",color="magenta",label="Loadings for PC4")
            plt.xticks(np.arange(0, 200, 10)) 
    #        plt.yticks(np.arange(0, 11, 1)) 
            plt.xlabel('Frequencies, THz', **axis_font)   
            plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
            plt.legend() 
            plt.tight_layout()
            plt.savefig('fig_loadings.png', dpi = 600)
            #plt.show()
            plt.close() 
        if 1:
# Savitsky-Golay filter            
            windows_size = 15
            polyorder = 2
            deriv = 0
            ind = ind / 1000000000000
            plt.plot(ind, savgol_filter(loadings[0,:], windows_size, polyorder, deriv),"-",color="blue",label="Loadings for PC1")
    #        plt.plot(ind, medfilt(loadings[1,:], 1),".",color="green",label="Loadings for PC1_1")
        #    plt.plot(ind, median_loadings_h,"--",color="black",label="median_loadings_h")
            plt.plot(ind, savgol_filter(loadings[1,:], windows_size, polyorder, deriv),"-",color="purple",label="Loadings for PC2")
        #    plt.plot(ind, median_loadings_il14,"-",color="blue",label="median_loadings_il14")
            #plt.plot(ind, savgol_filter(loadings[2,:], windows_size, polyorder, deriv),"-",color="red",label="Loadings for PC3")
            #plt.plot(ind, median_loadings_il28,"x",color="green",label="median_loadings_il28")
            #plt.plot(ind, savgol_filter(loadings[3,:], windows_size, polyorder, deriv),"-",color="magenta",label="Loadings for PC4")
            plt.axhline(y=np.sqrt(1/loadings[0].size), linestyle="--", color="black", label="Cutoff line")
            plt.axhline(y=-np.sqrt(1/loadings[0].size), linestyle="--", color="black")
            plt.xticks(np.arange(0, 2, 0.1)) 
    #        plt.yticks(np.arange(0, 11, 1)) 
            plt.xlabel('Frequencies, THz', **axis_font)   
            plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
            plt.legend() 
            plt.tight_layout()
            #plt.show()
            plt.savefig('thz_fig_loadings_not_normed.png', dpi = 600)
            #plt.show()
            plt.close()             
            with open ('max_loadings.bin', 'wb') as file:
                filtered_mat = np.abs(savgol_filter(loadings, windows_size, polyorder, deriv,axis=1))
                result_mat = [find_peaks(filtered_mat[i], height=0.02) for i in range(0,filtered_mat.shape[0])]
                pickle.dump(result_mat, file)                        
# Savitsky-Golay filter            
            windows_size = 15
            polyorder = 2
            deriv = 0
            plt.plot(ind, savgol_filter(loadings_normed[0,:], windows_size, polyorder, deriv),"-",color="blue",label="Loadings for PC1")
    #        plt.plot(ind, medfilt(loadings[1,:], 1),".",color="green",label="Loadings for PC1_1")
        #    plt.plot(ind, median_loadings_h,"--",color="black",label="median_loadings_h")
            plt.plot(ind, savgol_filter(loadings_normed[1,:], windows_size, polyorder, deriv),"-",color="purple",label="Loadings for PC2")
        #    plt.plot(ind, median_loadings_il14,"-",color="blue",label="median_loadings_il14")
            plt.plot(ind, savgol_filter(loadings_normed[2,:], windows_size, polyorder, deriv),"-",color="red",label="Loadings for PC3")
            #plt.plot(ind, median_loadings_il28,"x",color="green",label="median_loadings_il28")
            plt.plot(ind, savgol_filter(loadings_normed[3,:], windows_size, polyorder, deriv),"-",color="magenta",label="Loadings for PC4")
            plt.axhline(y=np.sqrt(np.sum(np.square(loadings_normed[0]))/loadings_normed[0].size), linestyle="--", color="blue", label="Cutoff line")
            plt.axhline(y=np.sqrt(np.sum(np.square(loadings_normed[1]))/loadings_normed[0].size), linestyle="--", color="purple")
            plt.axhline(y=np.sqrt(np.sum(np.square(loadings_normed[2]))/loadings_normed[0].size), linestyle="--", color="red")
            plt.axhline(y=np.sqrt(np.sum(np.square(loadings_normed[3]))/loadings_normed[0].size), linestyle="--", color="magenta")
            plt.axhline(y=-np.sqrt(np.sum(np.square(loadings_normed[0]))/loadings_normed[0].size), linestyle="--", color="blue")
            plt.axhline(y=-np.sqrt(np.sum(np.square(loadings_normed[1]))/loadings_normed[0].size), linestyle="--", color="purple")
            plt.axhline(y=-np.sqrt(np.sum(np.square(loadings_normed[2]))/loadings_normed[0].size), linestyle="--", color="red")
            plt.axhline(y=-np.sqrt(np.sum(np.square(loadings_normed[3]))/loadings_normed[0].size), linestyle="--", color="magenta")            
            plt.xticks(np.arange(0, 2, 0.1)) 
    #        plt.yticks(np.arange(0, 11, 1)) 
            plt.xlabel('Frequencies, THz', **axis_font)   
            plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
            plt.legend() 
            plt.tight_layout()
            plt.savefig('thz_fig_loadings_normed.png', dpi = 600)
            #plt.show()
            plt.close()      
    #        return 0    
    ##########################################################################################################################
    # Explained variance visualization         
    ##########################################################################################################################  
        if 0:        
            features = range(1, pipeline.named_steps.pca.n_components_ + 1)
            plt.bar(features, pipeline.named_steps.pca.explained_variance_ratio_, color='black')
            plt.xlabel('Principal Components')
            plt.ylabel('Explained variance')
            plt.xticks(features)
            l = brocken_stick(pipeline.named_steps.pca.n_components_)
            plt.plot(features, l)
            plt.tight_layout()
            plt.savefig('thz_fig_2_num_of_PC.png', dpi = 600)
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
                items_count = len(group1)
            elif i == 1:
                items_count = len(group2)
                #items_count = len(control_3)
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

        # Визуализация в файлы
        n_components = 10
        vis_folder_name = 'thz_visualization2d_' + str(n_components)
        if not os.path.exists(vis_folder_name):
            os.mkdir(vis_folder_name)
        for i in range(0, n_components):
            for j in range(i+1, n_components):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for group in splitted:
                    ax.scatter(group[0][:, i], group[0][:, j], s=256, alpha=0.8, c = group[1], label = group[2], marker=group[3])
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
        n_components_svm = 10
        pipeline=make_pipeline(StandardScaler(), PCA(n_components=n_components_svm))
        #pipeline=make_pipeline(PCA(n_components=4))
        #X1 = np.copy(X[len(U87_1):])
        XPCAreduced=pipeline.fit_transform(X)  

        num_features = len(frequencies)
        features = frequencies

        #X_sub = np.copy(XPCAreduced)
        X_sub = np.copy(X)

        # week 1
        #y_sub = np.array([2.7, 1.0, 2.9, 1.4, 0.7, 2.9, 2.9, 3.9, 2.9, 5.1 ])

        # week 2
        #y_sub = np.array([9.7, 10.0, 6.9, 5.8, 15.4, 6.4, 13.5, 11.7, 3.2, 23.0 ])
        # week 3
        #y_sub = np.array([68.1, 21.0, 139.2, 101.9, 141.8, 75.8, 60.1, 107.9, 86.5, 94.1])
        # week 4
        #y_sub = np.array([96.8, 51.9, 117.4, 110.9])
        # all
        y_sub = np.array([1.0, 2.9, 3.9, 2.9, 5.1, 
                        9.7, 10.0, 6.9, 5.8, 15.4, 6.4, 13.5, 11.7, 3.2, 23.0, 
                        21.0, 139.2, 101.9, 141.8, 75.8, 60.1, 107.9, 
                        #96.8, 51.9, 117.4, 110.9]
                        ])
        print("#############Lasso regression#############")

        random_state = np.random.RandomState(42)
        n_splits=10
        #cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
        #cv = RepeatedKFold(n_splits=n_splits, n_repeats=num_trials, random_state=random_state)
        #cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3)#, random_state=random_state)
        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        # define model
        model = LassoCV(alphas=np.arange(0, 1, 0.01), max_iter=100000, tol=0.01, cv=cv, n_jobs=-1)
        # fit model
        model.fit(X_sub, y_sub)
        # summarize chosen configuration
        print('alpha: %f' % model.alpha_)     
        alpha_optim = model.alpha_
        model = Lasso(alpha=alpha_optim, max_iter=100000, tol=0.01)
        # define model evaluation method
        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        # evaluate model
        scores = cross_val_score(model, X_sub, y_sub, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force scores to be positive
        scores = np.absolute(scores)
        #print(scores)
        print('Mean MAE: %.3f (%.3f)' % (np.mean(scores)/len(scores), np.std(scores)))        
        model.fit(X_sub, y_sub)
        print(model.score(X_sub, y_sub))
        coefficients = model.coef_
        importance = np.abs(coefficients)
        #print(np.array(features)[importance > 0] / 1000000000000)
        print(model.predict(X_sub[0].reshape(1,-1)))
        plt.plot(np.array(features / 1000000000000), importance)
        plt.xticks(np.arange(0, 2.2, 0.2)) 
        plt.xlabel('Frequencies, THz', **axis_font)   
        plt.ylabel('Lasso regression coefficients,\n Arb. Units', **axis_font)    
        plt.tight_layout()         
        plt.savefig('fig_thz_importance.png', dpi = 600)
        plt.close()        
        print('done')   


# #############################################################################
# Classification and ROC analysis for Linear SVM
# Run classifier with cross-validation and plot ROC curves
    n_components_svm = 10
    pipeline=make_pipeline(StandardScaler(), PCA(n_components=n_components_svm))
    #pipeline=make_pipeline(PCA(n_components=4))
    XPCAreduced=pipeline.fit_transform(X)  
    if 1:
        features = frequencies / 1000000000000
        num_features = len(features)
        #bound = len(U87_1) + len(control_1)
        X_sub = np.copy(X)
        if use_pca == 1:
            X_sub = np.copy(XPCAreduced)
            features = range(0,len(X_sub[0]))
            num_features = n_components_svm
        y_sub = np.copy(y)

        print("#############Linear SVM#############")
        random_state = np.random.RandomState(42)
        n_splits=10
        cv = StratifiedKFold(n_splits=n_splits)
        classifier = svm.SVC(kernel='linear', class_weight="balanced", probability=True,
                            random_state=random_state)

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
        print("STD Recall", np.std(recs, axis=0))
        print("Average Precision", np.mean(precs, axis=0))
        print("STD Precision", np.std(precs, axis=0))
        c_sum=np.zeros_like(confusion_matrices[0])
        for i in range (n_splits): 
            c_sum=c_sum+confusion_matrices[i]   
        print(np.mean(confusion_matrices,axis=0))
        print(np.std(confusion_matrices,axis=0))
        
        sn_sp_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            sn_sp_array[i][0], sn_sp_array[i][1]= evaluate_sp_sn(confusion_matrices[i])
        print('Mean and std of SN and SP')
        print(np.mean(sn_sp_array,axis=0))
        print(np.std(sn_sp_array,axis=0))       
        
        acc_prec_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            acc_prec_array[i][0], acc_prec_array[i][1]= evaluate_acc_prec(confusion_matrices[i])
        print('Mean and std of ACC and Prec')
        print(np.mean(acc_prec_array,axis=0))
        print(np.std(acc_prec_array,axis=0))               

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
        plt.savefig('fig_SVM_ROC.png', dpi = 600)
        plt.close()

        importances = np.mean(importances_ave, axis=0)
        sum_importances = np.sum(importances)
        importances /= sum_importances
        #indices = (importances).argsort()[::-1]   

        #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
        plt.bar(range(0,num_features), importances, color='b', align='center',linewidth=0 )
        if not use_pca:
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False)      # ticks along the bottom edge are off       
            #labelbottom=False # labels along the bottom edge are off
            
    #        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
            plt.xticks(range(0,240,30),np.around(np.arange(0,2.4,0.3),decimals=1))
            plt.xlabel('Frequencies, THz', **axis_font)   
        else:
            plt.xticks(range(0,len(importances),1),range(1,len(importances)+1,1))
            plt.xlabel('Principal component, number', **axis_font)   
        
        #plt.xlabel('Feature ID')
        plt.ylabel('Relative importance', **axis_font)
        plt.tight_layout()
        #plt.show()    
        plt.savefig('fig_importances_bars_SVM.png', dpi = 600)
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
        features = frequencies / 1000000000000
        #bound = len(U87_1) + len(control_1)
        num_features = len(features)        
        X_sub = np.copy(X)
        if use_pca == 1:
            X_sub = np.copy(XPCAreduced)
            features = range(0,len(X_sub[0]))
            num_features = n_components_svm
        y_sub = np.copy(y)


        print("#############Random forest#############")
        random_state = np.random.RandomState(42)
        n_splits=10
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

        sn_sp_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            sn_sp_array[i][0], sn_sp_array[i][1]= evaluate_sp_sn(confusion_matrices[i])
        print('Mean and std of SN and SP')
        print(np.mean(sn_sp_array,axis=0))
        print(np.std(sn_sp_array,axis=0))               

        acc_prec_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            acc_prec_array[i][0], acc_prec_array[i][1]= evaluate_acc_prec(confusion_matrices[i])
        print('Mean and std of ACC and Prec')
        print(np.mean(acc_prec_array,axis=0))
        print(np.std(acc_prec_array,axis=0))               

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
        plt.savefig('fig_RF_ROC.png', dpi = 600)
        #plt.show()
        plt.close()

        importances = np.mean(importances_ave, axis=0)
        sum_importances = np.sum(importances)
        importances /= sum_importances
        #indices = (importances).argsort()[::-1]   

        #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
        #plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        if not use_pca:
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False)      # ticks along the bottom edge are off       
            #labelbottom=False # labels along the bottom edge are off
            
    #        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
            plt.xticks(range(0,240,30),np.around(np.arange(0,2.4,0.3),decimals=1))
            plt.xlabel('Frequencies, THz', **axis_font)   
        else:
            plt.xticks(range(0,len(importances),1),range(1,len(importances)+1,1))
            plt.xlabel('Principal component, number', **axis_font)   
        
        #plt.xlabel('Feature ID')
        plt.ylabel('Relative importance', **axis_font)
        plt.tight_layout()
        #plt.show()    
        plt.savefig('fig_importances_bars_RF.png', dpi = 600)
        plt.close()

        ####### ROC for XGBoost################
    if 1:
        features = frequencies / 1000000000000
        #bound = len(U87_1) + len(control_1)
        num_features = len(features)        
        X_sub = np.copy(X)
        if use_pca == 1:
            X_sub = np.copy(XPCAreduced)
            features = range(0,len(X_sub[0]))
            num_features = n_components_svm
        y_sub = np.copy(y)


        print("#############XGBoost#############")
        random_state = np.random.RandomState(42)
        n_splits=10
        cv = StratifiedKFold(n_splits=n_splits)
        classifier = CatBoostClassifier(iterations=100, task_type="GPU", devices='0', learning_rate= 0.1, depth= 2)

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
            classifier.fit(X_sub[train], y_sub[train], verbose = 0)
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

        sn_sp_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            sn_sp_array[i][0], sn_sp_array[i][1]= evaluate_sp_sn(confusion_matrices[i])
        print('Mean and std of SN and SP')
        print(np.mean(sn_sp_array,axis=0))
        print(np.std(sn_sp_array,axis=0))               
        
        acc_prec_array = np.zeros((n_splits,2))
        for i in range(0,n_splits):
            acc_prec_array[i][0], acc_prec_array[i][1]= evaluate_acc_prec(confusion_matrices[i])
        print('Mean and std of ACC and Prec')
        print(np.mean(acc_prec_array,axis=0))
        print(np.std(acc_prec_array,axis=0))               

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
        plt.savefig('fig_xgboost_ROC.png', dpi = 600)
        #plt.show()
        plt.close()

        importances = np.mean(importances_ave, axis=0)
        sum_importances = np.sum(importances)
        importances /= sum_importances
        #indices = (importances).argsort()[::-1]   

        #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
        #plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        plt.bar(range(len(importances)), importances, color='b', align='center',linewidth=0 )
        if not use_pca:
            plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False)      # ticks along the bottom edge are off       
            #labelbottom=False # labels along the bottom edge are off
            
    #        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
            plt.xticks(range(0,240,30),np.around(np.arange(0,2.4,0.3),decimals=1))
            plt.xlabel('Frequencies, THz', **axis_font)   
        else:
            plt.xticks(range(0,len(importances),1),range(1,len(importances)+1,1))
            plt.xlabel('Principal component, number', **axis_font)   
        
        #plt.xlabel('Feature ID')
        plt.ylabel('Relative importance', **axis_font)
        plt.tight_layout()
        #plt.show()    
        plt.savefig('fig_importances_bars_XGboost.png', dpi = 600)
        plt.close()

    return 0

if __name__ == '__main__':
    main()