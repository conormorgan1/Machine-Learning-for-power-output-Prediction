import numpy as np
import pandas as pd
import sklearn as skl
import os, glob
import datetime as dt
from pathlib import Path
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from matplotlib import pyplot as plt
import timeit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor as knnr


testSelection = 17520
k=3


def main():
    temperatureValues()
    soltotFileConcatenation()
    dataframeConfiguration(air_temp_series, hourly_merged_df)
    pvPowerGeneration()
    svr_1()
    nnr_1()


def temperatureValues():
    temp_df = pd.read_csv("C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/hourlyTemperature.csv", low_memory=False)
    columns = ['date','temp']
    air_temp_df = temp_df[columns]
    air_temp_df2 = air_temp_df.iloc[223923:241443]
    global air_temp_series
    air_temp_series = air_temp_df2['temp']
    air_temp_series = pd.to_numeric(air_temp_series, errors='coerce')
    #print("Temp shape :", air_temp_series.shape[0])
    return air_temp_series


def soltotFileConcatenation():
    path = "C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/SoltotValues/"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    all_df = []
    my_file = Path("C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Datasets/featureData.csv")
    global hourly_merged_df

    if(my_file.exists()):
        hourly_merged_df = pd.read_csv(my_file)
    else:
        for f in all_files:
            df = pd.read_csv(f, sep=',', low_memory=False)
            all_df.append(df)
        merged_df = pd.concat(all_df, ignore_index=True, sort=True)
        columns = ['date','soltot']
        merged_df = merged_df[columns]
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.set_index('date')
        hourly_merged_df = merged_df.resample('H').mean()
        print("Merged:", hourly_merged_df.shape[0])

    return hourly_merged_df


def dataframeConfiguration(tempSeries, soltotDF):
    pd.to_numeric(tempSeries, errors='coerce')
    list = tempSeries.tolist()
    soltotDF['temp'] = list
    global featureDF
    featureDF = soltotDF
    featureDF.to_csv("C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Datasets/featureData.csv", index=False, encoding='utf-8')


def pvPowerGeneration():
    Pstc = 240
    Gstc = 1000
    K = 0.059
    Tr = 25

    pvList = []

    for index, row in featureDF.iterrows():
        Tc = row['temp']
        Ging = row['soltot']
        P = Pstc * (Ging / Gstc) * (1 + K * (Tc - Tr))
        if P < 0.0:
            P = P*-1
        pvList.append(P)

    pvSeries = pd.Series(pvList)
    featureDF['pv'] = pvList
    global completeDF
    completeDF = featureDF
    completeDF.drop(columns='Unnamed: 0')
    completeDF.reset_index()
    completeDF.to_csv("C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Datasets/completeDataSet.csv", index=False, encoding='utf-8')


def svr_1():
    completeDF['date'] = pd.to_datetime(completeDF['date'])
    labelsSeries = pd.Series(completeDF['pv'])
    features = completeDF[['soltot', 'temp']]
    vl = labelsSeries.iloc[:testSelection]
    vf = features.iloc[:testSelection]
    target = np.array(labelsSeries)
    data = np.array(features)

    scaler = StandardScaler()

    trainTiming = []
    predTiming = []

    mae_results = []
    mse_results = []
    r2_results = []
    validation_set_sizes=[]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.0014, shuffle=False)

    size_str =  "TRAIN INDEX SIZE: " + str(len(X_train)) + "\t\tTEST INDEX SIZE: " + str(len(X_test))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = GridSearchCV(svm.SVR(kernel='rbf', gamma='auto'),
                           param_grid={ "C": [1e0, 1e1, 1e2, 1e3],
                                       "epsilon": [1.0,.1,.01]},
                           cv = [(slice(None), slice(None))],
                           return_train_score=False,
                           n_jobs=-1)

    #regressor = svm.SVR(kernel='rbf', epsilon=.01, gamma='auto', C=100)

    start1 = timeit.default_timer()
    regressor.fit(X_train, y_train)
    ##param_df = pd.DataFrame(regressor.cv_results_)
    stop1 = timeit.default_timer()
    t1 = stop1 - start1
    print("SVM training time = ", t1, " seconds")
    start2 = timeit.default_timer()
    prediction = regressor.predict(X_test)
    stop2 = timeit.default_timer()
    t2 = stop2 - start2
    time_axes = range(len(prediction))
    plt.plot(time_axes, prediction, c='b', lw='0.7', label="Predicted")
    plt.plot(time_axes, y_test, c='r', lw='0.7', label="Actual")
    plt.legend(title='24 hour horizon: SVR')
    plt.savefig("SVR_24_plot.png", bbox_inches='tight')
    plt.show()

    mae_value = float("{:.4f}".format(metrics.mean_absolute_error(y_test, prediction)))
    mse_value = float("{:.6f}".format(metrics.mean_squared_error(y_test, prediction)))
    r2_value = float("{:.4f}".format(metrics.r2_score(y_test, prediction)))

    # mae_value = float("{:.4f}".format(mae_value))
    # mse_value = float("{:.4f}".format(mse_value))
    # r2_value = float("{:.4f}".format(r2_value))

    mae_results.append(mae_value)
    mse_results.append(mse_value)
    r2_results.append(r2_value)

    print("The model performance for testing set")
    print("————————————–————————————–————————————–")
    print("Mean Absolute Error = {}".format(mae_value))
    print("Mean Square Error = {}".format(mse_value))
    print("R-Squared score = {}".format(r2_value))
    print("————————————–————————————–————————————–")


    # totalTrainTime = np.sum(trainTiming)
    # avgTrainTime = totalTrainTime / k
    # maxTrainTime = np.max(trainTiming)
    # minTrainTime = np.min(trainTiming)

    #totalPredTime = np.sum(predTiming)
    # avgPredTime = totalPredTime / k
    # maxPredTime = np.max(predTiming)
    # minPredTime = np.min(predTiming)
    #
    totalProcessingTime = t1+t2

    best_Pars = [.01, 100]
    print("********************************************************************************************************")
    print("SUPPORT VECTOR REGRESSOR")
    # print("Parameters = ", best_Pars)
    # print("Total training time = ", totalTrainTime)
    # print("Max training time = ", maxTrainTime)
    # print("Min training time = ", minTrainTime)
    # print("Average training time = ", avgTrainTime)
    # print("********************************************************************************************************")
    # print("Total prediction time = ", totalPredTime)
    # print("Max prediction time = ", maxPredTime)
    # print("Min prediction time = ", minPredTime)
    # print("Average prediction time = ", avgPredTime)
    # print("********************************************************************************************************")
    print("Overall processing time (total training time + total prediction time) = ", totalProcessingTime)
    # print("********************************************************************************************************")
    #

    d = {'MAE': mae_results, 'MSE': mse_results , 'R2': r2_results}
    results = pd.DataFrame(data=d)
##param_df = param_df[['param_C', 'param_epsilon', 'mean_test_score']]
#results.to_csv('C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Results/results.txt', encoding='Utf-8', index=True, sep='\t')
    with open(r'C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Results/results.txt.txt', "w") as f:

        res_str = str(results.to_string)
        best_str = "Best fit parameters = C: " + str(best_Pars[1]) + " and epsilon: " + str(best_Pars[0])
        f.write("\n\n")
        f.write(best_str)
        f.write("\n\n")
        f.write(res_str)
        f.write("\n\n")
        f.write("\n Parameter results: ")
        f.write("\n\n")
        f.close()

def nnr_1():
    completeDF['date'] = pd.to_datetime(completeDF['date'])
    labelsSeries = pd.Series(completeDF['pv'])
    features = completeDF[['soltot', 'temp']]
    vl = labelsSeries.iloc[:testSelection]
    vf = features.iloc[:testSelection]
    target = np.array(labelsSeries)
    data = np.array(features)

    scaler = StandardScaler()

    trainTiming = []
    predTiming = []

    mae_results = []
    mse_results = []
    r2_results = []
    validation_set_sizes = []

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.0014, shuffle=False)

    size_str = "TRAIN INDEX SIZE: " + str(len(X_train)) + "\t\tTEST INDEX SIZE: " + str(len(X_test))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, shuffle=False)
    regressor = GridSearchCV(knnr(algorithm='auto'),
                            param_grid={ "weights": ['uniform', 'distance'],
                                        "n_neighbors": [3,5,7],
                                         "p": [1,2]},
                            cv = [(slice(None), slice(None))],
                            return_train_score=False,
                            n_jobs=-1)


    start1 = timeit.default_timer()
    regressor.fit(X_train, y_train)
    param_df = pd.DataFrame(regressor.cv_results_)
    stop1 = timeit.default_timer()
    t1 = stop1 - start1
    print("KNNR training time = ", t1, " seconds")
    start2 = timeit.default_timer()
    prediction = regressor.predict(X_test)
    stop2 = timeit.default_timer()
    t2 = stop2 - start2
    time_axes = range(0,len(prediction),1)
    plt.plot(time_axes, prediction, c='b', lw='0.7', label = "Predicted")
    plt.plot(time_axes, y_test, c='r', lw='0.7', label = "Actual")
    plt.legend(title="24 hour horizon: KNN")
    plt.savefig("KNNR_24_plot.png", bbox_inches='tight')
    plt.show()

    mae_value = float("{:.4f}".format(metrics.mean_absolute_error(y_test, prediction)))
    mse_value = float("{:.6f}".format(metrics.mean_squared_error(y_test, prediction)))
    r2_value = float("{:.4f}".format(metrics.r2_score(y_test, prediction)))

    # mae_value = float("{:.4f}".format(mae_value))
    # mse_value = float("{:.4f}".format(mse_value))
    # r2_value = float("{:.4f}".format(r2_value))

    mae_results.append(mae_value)
    mse_results.append(mse_value)
    r2_results.append(r2_value)

    print("The model performance for testing set")
    print("————————————–————————————–————————————–")
    print("Mean Absolute Error = {}".format(mae_value))
    print("Mean Square Error = {}".format(mse_value))
    print("R-Squared score = {}".format(r2_value))
    print("————————————–————————————–————————————–")


    trainTiming.append(t1)
    predTiming.append(t2)

    totalTrainTime = np.sum(trainTiming)
    avgTrainTime = totalTrainTime / k
    maxTrainTime = np.max(trainTiming)
    minTrainTime = np.min(trainTiming)

    totalPredTime = np.sum(predTiming)
    avgPredTime = totalPredTime / k
    maxPredTime = np.max(predTiming)
    minPredTime = np.min(predTiming)

    totalProcessingTime = totalPredTime + totalTrainTime

    best_Pars = regressor.best_params_
    print("********************************************************************************************************")
    print("K-Nearest Neihgbour Regression")
    print("Parameters = ", best_Pars)
    print("Total training time = ", totalTrainTime)
    print("Max training time = ", maxTrainTime)
    print("Min training time = ", minTrainTime)
    print("Average training time = ", avgTrainTime)
    print("********************************************************************************************************")
    print("Total prediction time = ", totalPredTime)
    print("Max prediction time = ", maxPredTime)
    print("Min prediction time = ", minPredTime)
    print("Average prediction time = ", avgPredTime)
    print("********************************************************************************************************")
    print("Overall processing time (total training time + total prediction time) = ", totalProcessingTime)
    print("********************************************************************************************************")


    d = {'MAE': mae_results, 'MSE': mse_results , 'R2': r2_results}
    results = pd.DataFrame(data=d)
    param_df = param_df[['param_p', 'param_weights','param_n_neighbors', 'mean_test_score']]
    #results.to_csv('C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Results/results.txt', encoding='Utf-8', index=True, sep='\t')
    with open(r'C:/Users/morga/Desktop/FINAL YEAR 2019/Project - Research Phase/ProjectData/Results/KNN_full_dataset/KNN_24_results.txt', "w") as f:
        x_i =1
        for s in validation_set_sizes:
            f.write("Cross Val Number (" + str(x_i) +")")
            f.write(s)
            f.write("\n\n")
            x_i += 1

        res_str = str(results.to_string)
        param_str = str(param_df.to_string)
        best_str = "Best fit parameters = p: " + str(best_Pars['p']) + ", weights: " + str(best_Pars['weights'] )+ "and n_neighbors: " + str(best_Pars['n_neighbors'])
        f.write("\n\n")
        f.write(best_str)
        f.write("\n\n")
        f.write(res_str)
        f.write("\n\n")
        f.write("\n Parameter results: ")
        f.write("\n\n")
        f.write( param_str )
        f.close()

main()
