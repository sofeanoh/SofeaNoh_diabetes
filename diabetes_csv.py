#%% 1. Imports
import os
import pickle #to save model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

#%% 2. Constants
CSV_PATH = os.path.join(os.getcwd(), 'datasets/diabetes.csv') #can do it this way or like the one in MODEL_PATH
print(CSV_PATH)

MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model.pkl')
######### the right way to do it (Cramer's V)
def cramers_corrected_stat(confusion_matrix):
    """
    
    This function calculate the Cramers V statistic for categorical-categorical assosication.
    Uses correction from Bergsma and wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328 
    """
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0] #take the confusion matrix and return the chi2
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

#%% 3. Data Loading
df = pd.read_csv(CSV_PATH)

#%% 4. EDA & Data inspection
print(df.head()) 
print("df shape:", df.shape)
print(df.info())
print(df.describe().transpose())

"""Is Pregnancies and Age categorical or continuous?
No one get pregnant 2.5 / 3.3 /4.5 so it is not continuous
For age, its controversial. if they do it by age-group then it is categorical
but if its values are numerical so it is continuous

why do we need to identify the type?

FOR PLOTTING PURPOSES ! :)

"""
#%% Data Visualisation
cat = ['Pregnancies', 'Outcome']
con = df.drop(labels=cat, axis=1).columns #will return a list of columns excluding the cat
print(con)
#for categorical data
for i in cat:
    plt.figure()
    sns.countplot(df[i])
    plt.show()
    
for i in con:
    plt.figure()
    sns.histplot(df[i], kde=True)
    plt.show()
    
df.boxplot()

'''
observation:

0s from the dataset might not always be outliers. therefore it is important to alway check
with the expert and profesionals from the domain. however, we can roughly
get the idea if the 0s make sense or not by plotting the data.

e.g:
0 level in glucose: no way people have 0 level
0 level of blood pressure: is this people die or what?
0 level of skin thickness: is this person made up of air?
0 level of insulin: no human ever have 0 in sulin
0 level of bmi: does this people live in the outer galaxy?

therefore we can infer that the cols with missing vals are:

[glucose, bp, skin thickness, insulin, bmi]

also, notice that the Insulin is right skewed. therefore there is bias.'''

# %% Recheck the 
#assume 0 = null values, now check any null values
cols_with_nulls = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' ]
for col in cols_with_nulls:
    sum_of_null_val = (df[col]==0).sum()
    print(col, ':', sum_of_null_val)

#%% 5. Data cleaning

#replace the 0 instances with NaN
for col in con: #can use col_with_nulls but since we already have con, and all cols_with_nulls are in con 
    df[col] = df[col].replace(0, np.nan) 
    
#now check the Null value occurences
print(df.isna().sum())

#yeay, now we have the null values! p/s: so now we can impute :)
#now if you re run the plotting, you will notice that the graph will be more normally distributed


#%% to fill in value for the NaNs / null

# 1. Method (directly fill in)

# to fill in using median value

# for col in cols_with_nulls:
#     df[col] = df[col].fillna(df[col].median())

#2. Method (using imputer)
# 1. simple imputer - mean, median, mode - when the missing value is less/not significant, or when the missing value doesnt really have any direct relationship/correlation with other features/mfgd
# 2. KNNImputer - it fills in an average value to the missing value after it scans the k nearest rows to the rows of the missing value.
# 3. IterativeImputer - your missing value has now become a target. it will find the correlation of other columns values to this target value, to predict what is the value to fill into this missing value
 ################################# IMPUTER ############################
# column_names = df.columns #to extract the column names
# knn_i = KNNImputer()
# df = knn_i.fit_transform(df) #this will return numpy array
# df = pd.DataFrame(df) #convert back into dataframe, but this will return no column names
# df.columns = column_names

#------------------------------Iterative imputer---------------------------------------------------
column_names = df.columns #to extract the column names
ii_i = IterativeImputer()
df = ii_i.fit_transform(df)
df = pd.DataFrame(df)
df.columns = column_names
print(df)

#check again if there is any null values and duplicates:
print(df.isna().sum())
print(df.duplicated().sum())
############################### END OF IMPUTING ###################################################
# %% 6. Feature Selection

#target: Outcome (categorical) 
#TIPS : look at the target column, if it is 
# To find correlation:
#for con vs categorical try to use logistic regression, 
#for cat and cat: use confusion matrix

#for con vs cat
for i in con:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=1), df['Outcome']) #another way to reshape. we can use to_frame(), add another [], or use this method, or even reshape
    score = lr.score(np.expand_dims(df[i], axis=1), df['Outcome'])
    print(i, score)
# %% 
#for cat vs cat

#categorical vs categorical data
matrix = pd.crosstab(df['Pregnancies'], df['Outcome']).to_numpy()
print(cramers_corrected_stat(matrix))

#%% Data Preprocessing and Splitting
#after we have identify the correlation, we can select which are gonna be our X 

#update1: remove 'skin thickness' and 'ddiabetes pedigree function' as it is not relevant from  user-end.
X = df.loc[:, con].drop(columns=['SkinThickness', 'DiabetesPedigreeFunction'], axis=1)  #since all col of con have high correlation to the label
y = df['Outcome']
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

# %% 8. Model Development ---> Pipeline
#we need to scale our data to: 1. improve model performance
#                              2. reduce outliers impact
#                              3. ensure data is on the same scale


#Standard scaler
#transform the data such that it has a mean of 0 and a std of 1
#less sensitive to outliers
#it assumes that our data is normally distributed
#pair best with algorithm that works with distance, eg. KNN, logistic, SVM
# can result in negative IF* the original value is below the mean value

#Min Max
#transform the data to a specific range, often between 0-1, by substracting the minimum value and divide
#by the range of 
#very sensitive to outliers
#does not distort the shape of distribution
#pair best with algorithm that requires input features to be within a bounded range, eg neural network
#can result in negative value IF* the data range is not fixed at [0,1]

##################### BEGINNING OF PIPELINE CREATIONS STEPS###################
#pipeline for standard scaler
pipeline_ss_lr = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('Logistic_Classifier', LogisticRegression())])
#pipeline for min max
pipeline_mms_lr = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('Logistic_Classifier', LogisticRegression())])

# ----------------------------------- Decision Tree -----------------------------
#pipeline for standard scaler, Decision Tree
pipeline_ss_dt = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('Decision_Tree', DecisionTreeClassifier())])


#pipeline for min max, Decision Tree
pipeline_mms_dt = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('Decision_Tree', DecisionTreeClassifier())])

# ------------------------------------Random Forest------------------------------
#pipeline for standard scaler, Random Forest
pipeline_ss_rf = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('Random_Forest', RandomForestClassifier())])


#pipeline for min max, Random Forest
pipeline_mms_rf = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('Random_Forest', RandomForestClassifier())])

#-------------------------------KNN Classifier-----------------------------------

#pipeline for standard scaler, KNNClassifier
pipeline_ss_knn = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('KNN', KNeighborsClassifier())])


#pipeline for min max, KNN classifier
pipeline_mms_knn = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('KNN', KNeighborsClassifier())])

#--------------------------------GradientBoosting-------------------------------

#pipeline for standard scaler, GradientBoosting
pipeline_ss_gb = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('GradientBoosting', GradientBoostingClassifier())])


#pipeline for min max, GradientBoosting classifier
pipeline_mms_gb = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('GradientBoosting', GradientBoostingClassifier())])

#-------------------------------Ada Boosting-----------------------------------

#pipeline for standard scaler, AdaBoosting
pipeline_ss_ab = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('AdaBoosting', AdaBoostClassifier())])


#pipeline for min max, AdaBoosting classifier
pipeline_mms_ab = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('AdaBoosting', AdaBoostClassifier())])

#--------------------------------ExtraTrees-------------------------------------

#pipeline for standard scaler, ExtraTrees
pipeline_ss_et = Pipeline([('Standard_Scaler', StandardScaler()),
                            ('ExtraTrees', ExtraTreesClassifier())])


#pipeline for min max, ExtraTrees classifier
pipeline_mms_et = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('ExtraTrees', ExtraTreesClassifier())])

######################### END OF PIPELINE CREATIONS STEP ####################

#To create a list to store all the pipeline
pipelines = [pipeline_ss_lr, pipeline_mms_lr, pipeline_ss_dt, pipeline_mms_dt, pipeline_ss_rf, pipeline_mms_rf, 
             pipeline_ss_knn, pipeline_mms_knn, pipeline_ss_gb, pipeline_mms_gb, pipeline_ss_ab, pipeline_mms_ab,
             pipeline_ss_et, pipeline_mms_et]

for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
pipe_scores=[]
for i, pipe in enumerate(pipelines):
    pipe_scores.append(pipe.score(X_test, y_test))
    
best_pipe = pipelines[np.argmax(pipe_scores)]
print("the best pipeline is", best_pipe)

best_pipe_score = pipe_scores[np.argmax(pipe_scores)]
print(best_pipe_score)

print('scores:' , pipe_scores)

# %% 9. Model Evaluation

#for categorical since the task is categorical (0 or 1)
y_pred = best_pipe.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)



# %%10. Hyperparameter tuning

# #pipeline for standard scaler
# #the best model
# pipeline_ss_lr = Pipeline([('Standard_Scaler', StandardScaler()),
#                             ('Logistic_Classifier', LogisticRegression())])

# grid_param = [{'Logistic_Classifier__C' : np.arange(0.0,2.0,0.1),
#                'Logistic_Classifier__intercept_scaling' : [1,5,10],
#                'Logistic_Classifier__solver' : ['lbfgs', 'liblinear', 'saga'],
#                'Logistic_Classifier__random_state': [42, 100]}]

# gridsearch = GridSearchCV(pipeline_ss_lr, param_grid=grid_param, cv=5, verbose=1)

################################UPDATE AFTER DROPPING TWO FEATURES#############
#after dropping SkinThickness and DiabetesPedigreeFunctions, the best model is no longer
#Logistic Regression with StandardScaler, instead with MinMaxScaler

#pipeline for min max
pipeline_mms_lr = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('Logistic_Classifier', LogisticRegression())])

grid_param = [{'Logistic_Classifier__C' : np.arange(0.0,2.0,0.01)}]

gridsearch = GridSearchCV(pipeline_mms_lr, param_grid=grid_param, cv=5, verbose=1)


grid = gridsearch.fit(X_train, y_train)

gridsearch.score(X_test, y_test)

print(grid.best_params_)
print(gridsearch.best_score_)

#%% 11. Best Model Selection
# best_model = grid.best_estimator_ #if decided to use from gridsearch, but it turns out that not using gridsearch yield better score
best_model = best_pipe

# %% 12. Save the model
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(best_model, file)

# %% END OF TRAINING
#not shuffling is better
#using iterative imputer is better
#not using gridsearchcv is better

# %%
