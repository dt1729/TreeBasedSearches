import numpy as np
from randomNums import random_nums
import matplotlib.pyplot as plt
def Predict(prevState, prevCov, stateMatrix, controlMatrix, ProcessCov, controlValues):
    stateEstimate = stateMatrix.dot(prevState) + controlMatrix.dot(controlValues)
    covEstimate = stateMatrix.dot(prevCov.dot(stateMatrix.T)) + ProcessCov
    return stateEstimate, covEstimate

def KalmanGain(prevCov, stateObservation, statePredict, covPredict, observationMatrix,ObservationCov):
    measurementRes = stateObservation - observationMatrix.dot(statePredict)
    covRes = observationMatrix.dot(covPredict.dot(observationMatrix.T)) + ObservationCov
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

    x = np.array([0,0])
    deltaT = 0.1
    scov = np.eye(2) # initial value of Process covariance matrix 
    scov[0][0] = 0.9 
    scov[0][1] = 0.1
    scov[1][0] = 0.2 
    scov[1][1] = 0.8

    stateMatrix = np.eye(2) # Ak
    controlMatrix = np.array([[deltaT,0],[0,deltaT]]) #Bk
    observationMatrix = np.eye(2) #Hk
    ObservationCov = np.array([[0.6,0.4],[0,1]]) # (R_k)
    ProcessCov = np.array([[1,0],[0,1]]) # process covariance matrix value (Q_k)

    filteredStates = []
    filteredCov = []

    a = np.array([np.array([i/10 +random_nums[i][0],i**2/100 + random_nums[i][1]]) for i in range(1,100)]) # this line adds noise to the exact model 
    actual = np.array([np.array([i/10,i**2/100]) for i in range(1,100)]) #exact model for plotting purposes

    print(a, "\n")
    prev = np.array([0,0])
    count = 0

    # todo controlValues, stateObservation, ObservationCov // Will be given inside a time loop
    for i in a:
        # controlValues = np.array([(a[count][0] - prev[0])/deltaT, (a[count][1] - prev[1])/deltaT])
        controlValues = np.array([1,0.1*(1+2*count)])
        x, scov = KalmanFilter(x,scov,controlValues,i,ProcessCov, stateMatrix, controlMatrix, observationMatrix)
        filteredStates.append(x)
        filteredCov.append(scov)
        # prev = i
        prev = x
        count = count + 1
        print(x)

    # Plotting of outputs    
    ansx = [i[0] for i in filteredStates]
    ansy = [i[1] for i in filteredStates]

    absx = [i[0] for i in a]
    absy = [i[1] for i in a]

    pathTrackedx = [i[0] for i in actual]
    pathTrackedy = [i[1] for i in actual]

    error_plotx = [a[i][0] - filteredStates[i][0] for i in range(len(a))]
    error_ploty = [a[i][1] - filteredStates[i][1] for i in range(len(a))]

    plt.figure()
    plt.subplot(121)
    plt.plot(ansx,ansy,'b', label='Filtered')
    plt.plot(absx,absy,'r', label='Measured')
    plt.plot(pathTrackedx, pathTrackedy, 'g', label='Mathematical Model')
    plt.title('Estimation')

    plt.subplot(122)
    plt.plot(error_plotx, label='Error in x')
    plt.plot(error_ploty, label='Error in y')
    plt.title('error')


    plt.legend()
    plt.show()
    




