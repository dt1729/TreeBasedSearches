import numpy as np

def Predict(prevState, prevCov, stateMatrix, controlMatrix, observationCov, controlValues):
    stateEstimate = stateMatrix.dot(prevState) + controlMatrix.dot(controlValues)
    covEstimate = stateMatrix.dot(prevCov.dot(stateMatrix.T)) + observationCov
    return stateEstimate, covEstimate

def KalmanGain(prevCov, stateObservation, statePredict, covPredict, observationMatrix):
    measurementRes = stateObservation - observationMatrix.dot(statePredict)
    covRes = observationMatrix.dot(covPredict.dot(observationMatrix.T)) # add R_k term
    # print(prevCov, "\n", covRes)
    KalGain = prevCov.dot((observationMatrix.T).dot(np.linalg.inv(covRes)))
    return KalGain, measurementRes

def Update(KalGain, statePredict, measurementRes, observationMatrix, covPredict):
    stateUpdate = statePredict + KalGain.dot(measurementRes)
    covUpdate  = (np.eye(observationMatrix.ndim)  - KalGain.dot(observationMatrix)).dot(prevCov)
    return stateUpdate, covUpdate

def KalmanFilter(prevState, prevCov, controlValues, stateObservation, observationCov, stateMatrix, controlMatrix, observationMatrix):
    statePredict, covPredict = Predict(prevState, prevCov, stateMatrix, controlMatrix, observationCov, controlValues) 
    # print("step1 done")
    KalGain, measurementRes = KalmanGain(prevCov, stateObservation,statePredict, covPredict, observationMatrix)
    # print("step2 done")
    stateUpdate, covUpdate = Update(KalGain, statePredict, measurementRes, observationMatrix,covPredict)
    # print("step3 done")
    return stateUpdate, covUpdate



if __name__ == "__main__":
    prevState = np.array([0,0])
    deltaT = 0.1
    prevCov = np.eye(2)
    prevCov[0][0] = 0.9 
    prevCov[0][1] = 0.1
    prevCov[1][0] = 0.2 
    prevCov[1][1] = 0.8
    stateMatrix = np.eye(2)
    controlMatrix = np.array([[deltaT,0],[0,deltaT]])
    observationMatrix = np.eye(2)
    ObservationCov = np.array([[0.5,0.5],[0.1,0.9]])
    filteredStates = []
    filteredCov = []
    a = np.array([np.array([i/10 +np.random.uniform(0,0.1),i/10 + np.random.uniform(0,0.1)]) for i in range(1,1000)])
    print(a)
    controlValues = np.array([1,1])
    # todo controlValues, stateObservation, ObservationCov // Will be given inside a time loop
    for i in a:
        prevState, prevCov = KalmanFilter(prevState,prevCov,controlValues,i,ObservationCov, stateMatrix, controlMatrix, observationMatrix)
        filteredStates.append(prevState)
        filteredCov.append(prevCov)
        print(prevState)



