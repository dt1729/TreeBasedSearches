import numpy as np
from randomNums import random_nums
def Predict(prevState, prevCov, stateMatrix, controlMatrix, ProcessCov, controlValues):
    stateEstimate = stateMatrix.dot(prevState) + controlMatrix.dot(controlValues)
    covEstimate = stateMatrix.dot(prevCov.dot(stateMatrix.T)) + ProcessCov
    return stateEstimate, covEstimate

def KalmanGain(prevCov, stateObservation, statePredict, covPredict, observationMatrix,ObservationCov):
    measurementRes = stateObservation - observationMatrix.dot(statePredict)
    covRes = observationMatrix.dot(covPredict.dot(observationMatrix.T)) + ObservationCov# add R_k term
    # print(prevCov, "\n", covRes)
    KalGain = prevCov.dot((observationMatrix.T).dot(np.linalg.inv(covRes)))
    return KalGain, measurementRes

def Update(KalGain, statePredict, measurementRes, observationMatrix, covPredict):
    stateUpdate = statePredict + KalGain.dot(measurementRes)
    covUpdate  = (np.eye(observationMatrix.ndim)  - KalGain.dot(observationMatrix)).dot(covPredict)
    return stateUpdate, covUpdate

def KalmanFilter(prevState, prevCov, controlValues, stateObservation, ProcessCov, stateMatrix, controlMatrix, observationMatrix):
    statePredict, covPredict = Predict(prevState, prevCov, stateMatrix, controlMatrix, ProcessCov, controlValues) 
    # print("step1 done")
    KalGain, measurementRes = KalmanGain(prevCov, stateObservation,statePredict, covPredict, observationMatrix, ObservationCov)
    # print("step2 done")
    stateUpdate, covUpdate = Update(KalGain, statePredict, measurementRes, observationMatrix,covPredict)
    # print("step3 done")
    return stateUpdate, covUpdate



if __name__ == "__main__":
    x = np.array([1,0])
    deltaT = 0.1
    scov = np.eye(2) # initial value of Process covariance matrix 
    scov[0][0] = 0.9 
    scov[0][1] = 0.1
    scov[1][0] = 0.2 
    scov[1][1] = 0.8
    stateMatrix = np.eye(2)
    controlMatrix = np.array([[deltaT,0],[0,deltaT]])
    observationMatrix = np.eye(2)
    ObservationCov = np.eye(2)
    ProcessCov = np.array([[0.99,0.01],[0.09,0.91]]) # process covariance matrix value
    filteredStates = []
    filteredCov = []
    a = np.array([np.array([i/10 +random_nums[i][0],i/100 + random_nums[i][1]]) for i in range(1,100)])
    # a = np.array([np.array([i/100,i/100]) for i in range(1,100)])
    print(a, "\n")
    controlValues = np.array([0.1,1])
    # todo controlValues, stateObservation, ObservationCov // Will be given inside a time loop
    for i in a:
        x, scov = KalmanFilter(x,scov,controlValues,i,ProcessCov, stateMatrix, controlMatrix, observationMatrix)
        filteredStates.append(x)
        filteredCov.append(scov)
        print(x)



