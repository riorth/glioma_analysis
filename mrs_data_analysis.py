import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from joblib import dump, load

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels) + 1
    mape = 100 * np.mean( errors / (test_labels + 1) - 1)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

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

###############################################################################################
#Read data from the source
###############################################################################################

#df_x = pd.read_excel ('neuro1_2.xlsx', sheet_name='X')
#df_y = pd.read_excel ('neuro1_2.xlsx', sheet_name='y')
#y= df_y["Class"].values

# df_x = pd.read_excel ('neuro1_2.xlsx', sheet_name='X_tract')
# df_y = pd.read_excel ('neuro1_2.xlsx', sheet_name='y_tract')
# y= df_y["Class"].values
# df_x.fillna(0, inplace=True)

# df_x = pd.read_excel ('neuro2_2.xlsx', sheet_name='X')
# df_y = pd.read_excel ('neuro2_2.xlsx', sheet_name='y')
# y= df_y["Class"].values
# df_x.fillna(0, inplace=True)

df_x = pd.read_excel ('mrs30102020.xlsx', sheet_name='lcmodel_inj')
df_y = pd.read_excel ('mrs30102020.xlsx', sheet_name='y_lcmodel_inj')
y= df_y["Group"].values
df_x.fillna(0, inplace=True)

# df_x2 = pd.read_excel ('neuro1_2.xlsx', sheet_name='X_tract2')
# df_x3 = pd.read_excel ('neuro1_2.xlsx', sheet_name='X_tract3')
# df_x2.fillna(0, inplace=True)
# df_x3.fillna(0, inplace=True)

X_tmp = df_x.values
names = list(df_x)

#X = X_tmp.copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_tmp)

num_features = 10
features = names
n_splits=3
#print('Covariance matrix', df_x2.corr())
# df_x3_t=df_x3.transpose()
# sn.heatmap(df_x3_t.corr(), annot=True, fmt='.1g')
# plt.show()
# plt.close()
###############################################################################################
#Prepare train and test datasets
###############################################################################################

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

###############################################################################################
#Train bagging classifier
###############################################################################################
if 0:
    print("#############Bagging SVM+RBF#############")
    #clf = BaggingClassifier(base_estimator=SVC(kernel='rbf'),
    #                        n_estimators=10, max_samples=0.7, random_state=0).fit(X_test, y_test)
    clf = BaggingClassifier(base_estimator=SVC(C=0.5, kernel='rbf', class_weight="balanced", gamma='scale'),
                             n_estimators=10, max_samples=1.0, random_state=0)                         
    random_state = np.random.RandomState(0)
    #n_splits=3
    cv = StratifiedKFold(n_splits=n_splits)
    classifier = clf

    tprs = []
    aucs = []
    precs = []
    recs = []
    confusion_matrices = []
    c_reps = []
    classification_reports = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                            name='ROC fold {}'.format(i),
#                            name='_nolegend_',
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        precs.append(precision_score(y[test], classifier.predict(X[test]),  average='weighted'))
        recs.append(recall_score(y[test], classifier.predict(X[test]),  average='weighted'))
        confusion_matrices.append(confusion_matrix(y[test], classifier.predict(X[test]), labels=[0, 1]))
        c_reps.append(classification_report(y[test], classifier.predict(X[test]), target_names=["Control", "Relapse"]))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)          
    print("Average Recall", np.mean(recs, axis=0))
    print("Average Precision", np.mean(precs, axis=0))
    c_sum=np.zeros_like(confusion_matrices[0])
    for i in range (n_splits): 
        c_sum=c_sum+confusion_matrices[i]   
    sn, sp = evaluate_sp_sn(c_sum)
    print(f'sensitivity= {sn:.2f}, specificity= {sp:.2f}') 
    acc, prec = evaluate_acc_prec(c_sum)
    print(f'accuracy= {acc:.2f}, precision= {prec:.2f}') 
    mean_tpr = np.mean(tprs, axis=0)    
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
    plt.show()
    plt.close()                 
    # print(np.mean(confusion_matrices,axis=0))
    # print(np.std(confusion_matrices,axis=0))
    if 0:    
    # Build step forward feature selection
        sfs_forward = sfs(clf,
            k_features=num_features,
            forward=True,
            floating=False,
            verbose=2,
            scoring='accuracy',
            cv=5,
            n_jobs=-1)

    # Perform SFFS
        sfs_forward = sfs_forward.fit(X_train, y_train, custom_feature_names=names)
    # Which features?
        feat_cols = list(sfs_forward.k_feature_idx_)
        print(feat_cols)
        print(sfs_forward.k_feature_names_)
        df1 = pd.DataFrame.from_dict(sfs_forward.get_metric_dict()).T
        df1.to_excel("output_sfs_forward.xlsx")

    # Build step backward feature selection
        sfs_backward = sfs(clf,
            k_features=num_features,
            forward=False,
            floating=False,
            verbose=2,
            scoring='accuracy',
            cv=5,
            n_jobs=-1)

    # # Perform SFFS
    #     sfs_backward = sfs_backward.fit(X_train, y_train, custom_feature_names=names)
    # # Which features?
    #     feat_cols = list(sfs_backward.k_feature_idx_)
    #     print(feat_cols)
    #     print(sfs_backward.k_feature_names_)
    #     df1 = pd.DataFrame.from_dict(sfs_backward.get_metric_dict()).T
    #     df1.to_excel("output_sfs_backward.xlsx")

    #     clf.fit(X_train, y_train)
    #     print('bagging classifier accuracy:', clf.score(X_test,y_test))


if 0:    
#    dump(clf, 'BaggingClassifier.joblib') 

    # clf1 = DecisionTreeClassifier(max_depth=4)
    # clf2 = SVC(kernel='rbf', probability=True)
    # eclf = VotingClassifier(
    #     estimators=[('dt', clf1), ('svm', clf2)],
    #     voting='soft', weights=[1, 2])


    # for clf, label in zip([clf1, clf2, eclf], ['DT', 'SVM', 'Ensemble']):
    #     scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    clf3 = SVC(kernel='linear', class_weight="balanced").fit(X_train, y_train)
    #dump(clf3, 'svm.joblib') 
    scores = cross_val_score(clf3, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    print("Score: %0.2f" % clf3.score(X_test,y_test))
    features = names
    importances = abs(clf3.coef_[0])
    sum_importances = np.sum(importances)
    importances /= sum_importances
    indices = (importances).argsort()[::-1]
    y_pred = clf3.predict(X_test)
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    sn, sp = evaluate_sp_sn(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(f'sensitivy= {sn}, specificity= {sp}')
    print(classification_report(y_test, y_pred, target_names=["Control", "Relapse"]))
            
    plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center')
    plt.xticks(range(len(indices[:num_features])), [features[i] for i in indices[:num_features]])
    plt.xlabel('Признаковые переменные')
    plt.ylabel('Относительная важность')
    plt.show()    
    plt.close()
    #plt.title('')
    #disp = plot_precision_recall_curve(clf3, X_test, y_test)
    #disp.ax_.set_title('2-class Precision-Recall curve: '
    #                'AP={0:0.2f}'.format(average_precision))    
# #############################################################################
# Classification and ROC analysis for Linear SVM
# Run classifier with cross-validation and plot ROC curves
if 1:
    print("#############Linear SVM#############")
    random_state = np.random.RandomState(0)
    #n_splits=3
    cv = StratifiedKFold(n_splits=n_splits)
    classifier = SVC(kernel='linear', class_weight="balanced", probability=True,
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
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                            name='ROC fold {}'.format(i),
#                            name='_nolegend_',
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        precs.append(precision_score(y[test], classifier.predict(X[test]),  average='weighted'))
        recs.append(recall_score(y[test], classifier.predict(X[test]),  average='weighted'))
        confusion_matrices.append(confusion_matrix(y[test], classifier.predict(X[test]), labels=[0, 1]))
        c_reps.append(classification_report(y[test], classifier.predict(X[test]), target_names=["Control", "Ill"]))
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
        title="Linear SVM ROC-analysis")
    ax.legend(loc="lower right")
    plt.show()
    plt.close()

    importances = np.mean(importances_ave, axis=0)
    sum_importances = np.sum(importances)
    importances /= sum_importances
    indices = (importances).argsort()[::-1]   

    plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center')
    plt.xticks(range(len(indices[:num_features])), [features[i] for i in indices[:num_features]])
    plt.xlabel('Признаковые переменные')
    plt.ylabel('Относительная важность')
    plt.show()    
    plt.close()


###############################################################################################
# Create the parameter grid based on the results of random search 
###############################################################################################
if 0:
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [2, 4, 8, 12],
        'n_estimators': [10, 50, 100, 200, 300, 1000]
    }

    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 5, n_jobs = -1, verbose = 2)

    base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_test, y_test)

    #print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
###############################################################################################
#Working with Random forest classifier
###############################################################################################
if 1:
#for num_runs in range(20):
    print("#############Random forest#############")
#    random_state = np.random.RandomState(0)
    #n_splits=3
    cv = StratifiedKFold(n_splits=n_splits)
    #classifier = RandomForestClassifier(n_estimators=best_grid.n_estimators, bootstrap = best_grid.bootstrap, max_depth= best_grid.max_depth, 
    #                        min_samples_leaf=best_grid.min_samples_leaf, min_samples_split=best_grid.min_samples_split)
    classifier = RandomForestClassifier(n_estimators=10, bootstrap = True, max_depth= 110, min_samples_leaf=2, min_samples_split=4)
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
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                            name='ROC fold {}'.format(i),
#                            name='_nolegend_',
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        precs.append(precision_score(y[test], classifier.predict(X[test]),  average='weighted'))
        recs.append(recall_score(y[test], classifier.predict(X[test]),  average='weighted'))
        confusion_matrices.append(confusion_matrix(y[test], classifier.predict(X[test]), labels=[0, 1]))
        c_reps.append(classification_report(y[test], classifier.predict(X[test]), target_names=["Control", "Ill"]))
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
        title="ROC-анализ для классификатора случайный лес")
    ax.legend(loc="lower right")
    plt.show()
    plt.close()

    importances = np.mean(importances_ave, axis=0)
    sum_importances = np.sum(importances)
    importances /= sum_importances
    indices = (importances).argsort()[::-1]   

    plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center')
    plt.xticks(range(len(indices[:num_features])), [features[i] for i in indices[:num_features]])
    plt.xlabel('Признаковые переменные')
    plt.ylabel('Относительная важность')
    plt.show()    
    plt.close()


# clf2 = RandomForestClassifier(n_estimators=best_grid.n_estimators, bootstrap = best_grid.bootstrap, max_depth= best_grid.max_depth, 
#                             min_samples_leaf=best_grid.min_samples_leaf, min_samples_split=best_grid.min_samples_split)

# clf2.fit(X,y)
# dump(clf2, 'random_forest.joblib') 
# scores = cross_val_score(clf2, X, y, scoring='accuracy', cv=50)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
# print("Score: %0.2f" % clf2.score(X_test,y_test))

for name, importance in zip(names, clf2.feature_importances_):
    print(name, "=", importance)

features = names
importances = clf2.feature_importances_
indices = (importances).argsort()[::-1]
plt.bar(range(len(indices[:num_features])), importances[indices[:num_features]], color='b', align='center')
plt.xticks(range(len(indices[:num_features])), [features[i] for i in indices[:num_features]])
plt.xlabel('Признаковые переменные')
plt.ylabel('Относительная важность')
plt.show()    

#Check age distribution
# X_age_corean = X[np.where(y==2)]
# X_age_slavic = X[np.where(y==1)]
# X_age_corean[:,2].mean()
# X_age_slavic[:,2].mean()

clf2_disp = metrics.plot_roc_curve(clf2, X_test, y_test)
print(clf2.score(X,y))
plt.show()
###############################################################################################
#Done Working with Random forest classifier
###############################################################################################

print("done")


