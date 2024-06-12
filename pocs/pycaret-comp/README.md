### Result
* pycaret
* compare models
* AI / MLOps

Result:
```
❯ ./run.sh
   Number of times pregnant  Plasma glucose concentration a 2 hours in an oral glucose tolerance test  ...  Age (years)  Class variable
0                         6                                                148                         ...           50               1
1                         1                                                 85                         ...           31               0
2                         8                                                183                         ...           32               1
3                         1                                                 89                         ...           21               0
4                         0                                                137                         ...           33               1

[5 rows x 9 columns]
                    Description             Value
0                    Session id               123
1                        Target    Class variable
2                   Target type            Binary
3           Original data shape          (768, 9)
4        Transformed data shape          (768, 9)
5   Transformed train set shape          (537, 9)
6    Transformed test set shape          (231, 9)
7              Numeric features                 8
8                    Preprocess              True
9               Imputation type            simple
10           Numeric imputation              mean
11       Categorical imputation              mode
12               Fold Generator   StratifiedKFold
13                  Fold Number                10
14                     CPU Jobs                -1
15                      Use GPU             False
16               Log Experiment             False
17              Experiment Name  clf-default-name
18                          USI              4372
```
```
                                    Model  Accuracy     AUC  Recall   Prec.  \                                                                                               
lr                    Logistic Regression    0.7689  0.8047  0.5602  0.7208   
ridge                    Ridge Classifier    0.7670  0.8060  0.5497  0.7235   
lda          Linear Discriminant Analysis    0.7670  0.8055  0.5550  0.7202   
rf               Random Forest Classifier    0.7485  0.7911  0.5284  0.6811   
nb                            Naive Bayes    0.7427  0.7955  0.5702  0.6543   
gbc          Gradient Boosting Classifier    0.7373  0.7909  0.5550  0.6445   
ada                  Ada Boost Classifier    0.7372  0.7799  0.5275  0.6585   
et                 Extra Trees Classifier    0.7299  0.7788  0.4965  0.6516   
qda       Quadratic Discriminant Analysis    0.7282  0.7894  0.5281  0.6558   
lightgbm  Light Gradient Boosting Machine    0.7133  0.7645  0.5398  0.6036   
knn                K Neighbors Classifier    0.7001  0.7164  0.5020  0.5982   
dt               Decision Tree Classifier    0.6928  0.6512  0.5137  0.5636   
xgboost         Extreme Gradient Boosting    0.6891  0.7572  0.5292  0.5668   
dummy                    Dummy Classifier    0.6518  0.5000  0.0000  0.0000   
svm                   SVM - Linear Kernel    0.5954  0.5914  0.3395  0.4090   

              F1   Kappa     MCC  TT (Sec)  
lr        0.6279  0.4641  0.4736     1.683  
ridge     0.6221  0.4581  0.4690     0.013  
lda       0.6243  0.4594  0.4695     0.011  
rf        0.5924  0.4150  0.4238     0.082  
nb        0.6043  0.4156  0.4215     0.015  
gbc       0.5931  0.4013  0.4059     0.063  
ada       0.5796  0.3926  0.4017     0.050  
et        0.5596  0.3706  0.3802     0.071  
qda       0.5736  0.3785  0.3910     0.012  
lightgbm  0.5650  0.3534  0.3580    75.749  
knn       0.5413  0.3209  0.3271     0.255  
dt        0.5328  0.3070  0.3098     0.015  
xgboost   0.5438  0.3089  0.3122     0.091  
dummy     0.0000  0.0000  0.0000     0.017  
svm       0.2671  0.0720  0.0912     0.015  
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

```