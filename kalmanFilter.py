import numpy as np

def KalmanFilter(prevState,prevCov, stateObservation, ObservationCov, stateMatrix, controlMatrix, observationMatrix):
    controlValues = np.ones(0,0)
    statePredict, covPredict = Predict(prevState, prevCov, stateMatrix, controlMatrix, ObservationCov, controlValues) 
    KalGain, measurementRes = KalmanGain(prevCov, stateObservation,statePredict, covPredict, observationMatrix)
    stateUpdate, covUpdate = Update(KalGain, statePredict, measurementRes, observationMatrix,covPredict)
    return stateUpdate, covUpdate

def Predict(prevState, prevCov, stateMatrix, controlMatrix, ObservationCov, controlValues):
    stateEstimate = stateMatrix.dot(prevState) + controlMatrix.dot(controlValues)
    covEstimate = stateMatrix.dot(prevCov.dot(stateMatrix.T)) + ObservationCov
    return stateEstimate, covEstimate

def Update(KalGain, statePredict, measurementRes, observationMatrix, covPredict):
    stateUpdate = statePredict + KalGain.dot(measurementRes)
    covUpdate  = (np.eye(3)  - KalGain.dot(observationMatrix)).dot(prevCov)
    pass

def KalmanGain(prevCov, stateObservation, statePredict, covPredict, observationMatrix):
    measurementRes = stateObservation - observationMatrix.dot(statePredict)
    covRes = observationMatrix.dot(prevCov.dot(observationMatrix.T)) + R_k
    KalGain = prevCov.dot((observationMatrix.T).dot(np.linalg.inv(covRes))
    return KalGain, measurementRes