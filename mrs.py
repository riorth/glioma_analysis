##%%
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
from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold
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
import seaborn as sns
from matplotlib.pyplot import figure



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


#def main():
axis_font = {'fontname':'Arial'}
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (16,9)
col = ['Red', 'Green', 'Blue', 'Yellow', 'Black']
print('start')
##%%
if 1: 
#############################################################################################################
    #msu data analysis
#############################################################################################################     
    path_to_U87_1 = os.path.join('data/U87_1/')
    path_to_U87_2 = os.path.join('data/U87_2/')
    path_to_U87_3 = os.path.join('data/U87_3/')
    path_to_U87_4 = os.path.join('data/U87_4/')
    path_to_control_1 = os.path.join('data/Control_1/')
    path_to_control_2 = os.path.join('data/Control_2/')
    path_to_control_3 = os.path.join('data/Control_3/')
    path_to_control_4 = os.path.join('data/Control_4/')             


    frequencies = []
    U87_1 = []
    U87_2 = []
    U87_3 = []
    U87_4 = []
    control_1 = []
    control_2 = []
    control_3 = []
    control_4 = []


    # with open(frequency_path_msu) as f:
    #     lines = f.readlines()
    # for line in lines:
    #     line.replace('\n','')
    #     frequencies_msu.append(float(line))
#############################################################################################################    
    # Чтение данных из файлов
#############################################################################################################                                                                           
# Read MRS data
    #df_x1 = pd.read_excel('mrs30102020.xlsx', sheet_name='LCModel Concentration')
    df_x1 = pd.read_excel('mrs30102020.xlsx', sheet_name='LCModel SD')
    #df_x1.fillna(0, inplace=True)LCModel SD
    X = np.asarray(df_x1.values)
    df_y =  pd.read_excel('mrs30102020.xlsx', sheet_name='y LCModel Concentration')
    y_g = np.asarray(df_y["Group"].values)
    y = np.asarray(df_y["Class"].values)
    X1 = X[:69]  
    y1 =  y[:69]
    y1_g =  y_g[:69]
    # X1 = X[69:]    
    # y1 =  y[69:]
    # y1_g =  y_g[69:]

    unique, counts = np.unique(y1_g, return_counts=True)
    labels_dict = dict(zip(unique, counts))
    labels_names = list(unique)
# Select classes
    #labels_dict["Control_1"]  
    mark1_name = 'Tumor_1'
    mark2_name = 'Control_1'
    mark1 = 0
    mark2 = 1
    img_path = 'd:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\mrs\\3week\\sd\\'    
    #img_path = 'd:\\Work\\Science\\projects\\_china2020\\_ICLO2020\\mrs\\3week\\sd\\u87\\'    
    # controlateral Tumor_3 contains 9 samples!
    skip_samples = 40
    num_samples = 19
    X = np.copy(X1[skip_samples:skip_samples+num_samples])
    y = np.copy(y1[skip_samples:skip_samples+num_samples])
    labels = np.copy(y1_g[skip_samples:skip_samples+num_samples])


    df_x2 = pd.read_excel('mrs30102020.xlsx', sheet_name='LCModel related to Cr+PCr')
    names = list(df_x1)
    #df_x2.fillna(0, inplace=True)        

    print ('done files reading')  

#############################################################################################################
    # Формируем массив данных
#############################################################################################################    
    u87_2len = len(U87_1)+len(U87_2)
    u87_3len = len(U87_1)+len(U87_2) + len(U87_3)
    u87_4len = len(U87_1)+len(U87_2)+len(U87_3)+len(U87_4)        
    ctrl_2len = len(control_1)+len(control_2)
    ctrl_3len = len(control_1)+len(control_2)+len(control_3)        
    ctrl_4len = len(control_1)+len(control_2)+len(control_3) + len(control_4)        
############## Mean and median spectra #############################
    if 0:        
                    
        plt.plot(np.mean(X[:labels_dict["Tumor_1"]], axis=0),color='green', label="mean Tumor_1")          
        plt.plot(np.mean(X[labels_dict["Tumor_1"]:labels_dict["Tumor_1"]+labels_dict["Control_1"]], axis=0),color='blue', label="mean Control_1")
        # plt.plot(np.mean(X[u87_2len:u87_3len], axis=0),color='red', label="mean U87_3")
        # plt.plot(np.mean(X[u87_3len:u87_4len], axis=0),color='magenta', label="mean U87_4")

        plt.legend()
        plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)
        plt.ylabel('Raman intensities, Arb. Units', **axis_font)    
        plt.tight_layout()           
        plt.show()
        plt.close()

###################################################################
    pipeline=make_pipeline(StandardScaler(), PCA(n_components=10))
    #pipeline=make_pipeline(PCA(n_components=10))
    XPCAreduced=pipeline.fit_transform(X) 
    plt.plot(np.cumsum((pipeline.named_steps.pca.explained_variance_ratio_)))
    plt.tight_layout()
    plt.savefig(img_path+'explained_variance.png', dpi = 600)
    plt.close()     
    #pipeline=make_pipeline(StandardScaler(), PCA(n_components=4))
    #XPCAreduced=pipeline.fit_transform(X)             
    #plt.show()  
    n_components = 4
    loadings = np.zeros(pipeline.named_steps.pca.components_.shape)
    for i in range(len(pipeline.named_steps.pca.components_)):
        loadings[i,:] = pipeline.named_steps.pca.components_[i,:] * np.sqrt(pipeline.named_steps.pca.explained_variance_)[i]
    ind = range(len(names))

    random_state = np.random.RandomState(14)   

    if 1:   
        corr = np.corrcoef(X)
        ax=sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        plt.savefig(img_path+'correlation.png', dpi = 600)
        plt.close()


    print ('Perform SFFS')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = random_state, stratify=y)
    clf = svm.SVC(kernel='linear', class_weight="balanced", probability=True,
                    random_state=random_state)

    # Build step forward feature selection
    if 0:
        sfs_forward = sfs(clf,
            k_features=(2,10),
            forward=True,
            floating=False,
            verbose=0,
            scoring='accuracy',
            cv=5,
            n_jobs=-1)                    
        sfs_forward = sfs_forward.fit(X_train, y_train, custom_feature_names=names)
    # Which features?
        feat_cols = list(sfs_forward.k_feature_idx_)
        print('best combination (ACC: %.3f): %s\n' % (sfs_forward.k_score_, sfs_forward.k_feature_idx_))
        print('all subsets:\n', sfs_forward.subsets_)
        plot_sfs(sfs_forward.get_metric_dict(), kind='std_err')      
        print(feat_cols)
        print(sfs_forward.k_feature_names_)

# Build step backward feature selection
    if 0:
        sfs_backward = sfs(clf,
            k_features=(2,10),
            forward=False,
            floating=False,
            verbose=0,
            scoring='accuracy',
            cv=5,
            n_jobs=-1)
    
# Perform SFFS
        sfs_backward = sfs_backward.fit(X_train, y_train, custom_feature_names=names)
# Which features?
        feat_cols = list(sfs_backward.k_feature_idx_)
        print(feat_cols)
        print(sfs_backward.k_feature_names_)

        clf.fit(X_train, y_train)
        print('classifier accuracy:', clf.score(X_test,y_test))



    rfe = RFECV(clf, step=1, cv=5)
    fit = rfe.fit(X, y)
    print("RFECV linear svm")
    print("Optimal number of features : %d" % rfe.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.tight_layout()
    plt.savefig(img_path+'cv_scores.png', dpi = 600)
    plt.close()                                                                      
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))

    clf = RandomForestClassifier(n_estimators=100, bootstrap = True)
    rfe = RFECV(clf, step=1, cv=5)
    fit = rfe.fit(X, y)
    print("RFECV random forest")    
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))    

    ##################################

##########################################################################################################################
# Loadings visualization
##########################################################################################################################  
    if 1:
        plt.plot(ind, loadings[0,:],"-.",color="blue",label="Loadings for PC1")
#        plt.plot(ind, medfilt(loadings[1,:], 1),".",color="green",label="Loadings for PC1_1")
    #    plt.plot(ind, median_loadings_h,"--",color="black",label="median_loadings_h")
        plt.plot(ind, loadings[1,:],":",color="purple",label="Loadings for PC2")
    #    plt.plot(ind, median_loadings_il14,"-",color="blue",label="median_loadings_il14")
        plt.plot(ind, loadings[2,:],"-",color="red",label="Loadings for PC3")
        #plt.plot(ind, median_loadings_il28,"x",color="green",label="median_loadings_il28")
        plt.plot(ind, loadings[3,:],"X",color="magenta",label="Loadings for PC4")
        plt.xticks(range(len(names))) 
#        plt.yticks(np.arange(0, 11, 1)) 
        plt.xlabel('Metabolites ID', **axis_font)
        plt.ylabel('PCA Loadings values, a.u.', **axis_font)   
        plt.legend() 
        plt.tight_layout()
        #plt.savefig('fig_loadings.png', dpi = 600)
        #plt.show()
        plt.savefig(img_path+'fig_loadings.png', dpi = 600)
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
        plt.savefig(img_path+'explained_variance_ratio.png', dpi = 600)
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
            items_count = int(labels_dict[mark1_name])
        elif i == 1:
#                items_count = len(U87_4)
            items_count = int(labels_dict[mark2_name])
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
        splitted.append((XPCAreduced[items_to_skip:items_to_skip + items_count], col[i], labels[items_to_skip+1], markers[i]))
        items_to_skip += items_count

    # Визуализация в файлы
    vis_folder_name = img_path + 'raman_visualization2d_' + str(n_components)
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

# #############################################################################
# Classification and ROC analysis for Linear SVM
# Run classifier with cross-validation and plot ROC curves

## %%
if 1:
    # pipeline=make_pipeline(StandardScaler())
    # X_scaled=pipeline.fit_transform(X)
    num_features = len(names)
    features = names
    #bound = len(U87_1) + len(control_1)
    #X_sub = XPCAreduced[:bound]
    #y_sub = np.asarray(y[:bound])
    X_sub = np.copy(X)
    y_sub = np.copy(y)

    print("#############Linear SVM#############")
    rs = 42
    n_splits=3
    #cv = StratifiedKFold(n_splits=n_splits, shuffle= True)
    cv = StratifiedKFold(n_splits=n_splits, random_state=rs)
    classifier = svm.SVC(kernel='linear', class_weight="balanced", probability=True, random_state=rs)
    #classifier = svm.SVC(kernel='linear', class_weight="balanced", probability=True)

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
        confusion_matrices.append(confusion_matrix(y_sub[test], classifier.predict(X_sub[test])))
        c_reps.append(classification_report(y_sub[test], classifier.predict(X_sub[test])))
        #print(classification_report(y_sub[test], classifier.predict(X_sub[test]), target_names=[mark1, mark2]))
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
    plt.savefig(img_path+'roc_svm.png', dpi=600)
    #plt.show()
    plt.close()

    importances = np.mean(importances_ave, axis=0)
    sum_importances = np.sum(importances)
    importances /= sum_importances
    #indices = (importances).argsort()[::-1]   

    plt.figure(figsize=(10,8))
    #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
    plt.barh(range(len(importances)), importances, color='b', align='center',linewidth=0 )
    

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False)      # ticks along the bottom edge are off       
    #labelbottom=False # labels along the bottom edge are off
    plt.tick_params(axis="y",direction="in", pad=-600, labelleft=True, labelright=False )
#        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
    plt.yticks(range(0,20,1), names)
    #plt.axis["left"].major_ticklabels.set_ha("left")
    #plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)   
    #plt.xlabel('Feature ID')
    plt.xlabel('Relative importance', **axis_font)
    plt.tight_layout()
    #plt.show()    
    plt.savefig(img_path+'importance_svm.png', dpi=600)
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
## %%
####### ROC for random forest################
if 1:
    # pipeline=make_pipeline(StandardScaler())
    # X_scaled=pipeline.fit_transform(X)
    num_features = len(names)
    features = names
    #bound = len(U87_1) + len(control_1)
    #X_sub = XPCAreduced[:bound]
    #y_sub = np.asarray(y[:bound])
    X_sub = np.copy(X)
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
        confusion_matrices.append(confusion_matrix(y_sub[test], classifier.predict(X_sub[test])))
        c_reps.append(classification_report(y_sub[test], classifier.predict(X_sub[test])))
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
    plt.savefig(img_path+'roc_rf.png', dpi=600)
    plt.close()

    importances = np.mean(importances_ave, axis=0)
    sum_importances = np.sum(importances)
    importances /= sum_importances
    #indices = (importances).argsort()[::-1]   

    plt.figure(figsize=(10,8))
    #plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center',linewidth=0 )
    plt.barh(range(len(importances)), importances, color='b', align='center',linewidth=0 )
    

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False)      # ticks along the bottom edge are off       
    #labelbottom=False # labels along the bottom edge are off
    plt.tick_params(axis="y",direction="in", pad=-600, labelleft=True, labelright=False )
#        plt.xticks(range(num_features), ['%.2f'%(features[i]) for i in indices[:num_features]])
    plt.yticks(range(0,20,1), names)
    #plt.axis["left"].major_ticklabels.set_ha("left")
    #plt.xlabel('Raman shift, cm$^{-1}$', **axis_font)   
    #plt.xlabel('Feature ID')
    plt.xlabel('Relative importance', **axis_font)
    plt.tight_layout()
    #plt.show()    
    plt.savefig(img_path+'importance_rf.png', dpi=600)
    plt.close()


###############################################################################################
#Done Working with Random forest classifier
###############################################################################################

#     return 0

# if __name__ == '__main__':
#     main()

## %%
