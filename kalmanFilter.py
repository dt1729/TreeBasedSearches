import numpy as np
from randomNums import random_nums
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self,prevState, prevCov, ProcessCov, stateMatrix, controlMatrix, observationMatrix,ObservationCov):
        self.x = prevState #evolving from apriori state estimate
        self.prevCov = prevCov #evolving from previous estimate
        self.Q = ProcessCov 
        self.A = stateMatrix
        self.B = controlMatrix
        self.H = observationMatrix
        self.R = ObservationCov

    def Predict(self,controlValues):
        stateEstimate = self.A.dot(self.x) + self.B.dot(controlValues)
        covEstimate = self.A.dot(self.prevCov.dot(self.A.T)) + self.Q
        return stateEstimate, covEstimate

    def KalmanGain(self,xhat, Pk, stateObservation):
        measurementRes = stateObservation - self.H.dot(xhat)
        covRes = self.H.dot(Pk.dot(self.H.T)) + self.R
        KalGain = self.prevCov.dot((self.H.T).dot(np.linalg.inv(covRes)))
        return KalGain, measurementRes

    def Update(self,KalGain, xhat, measurementRes, Pk):
        stateUpdate = xhat + KalGain.dot(measurementRes)
        covUpdate  = (np.eye(self.H.ndim)  - KalGain.dot(self.H)).dot(Pk)
        return stateUpdate, covUpdate

    def KF(self,controlValues,stateObservation):
        statePredict, covPredict = self.Predict(controlValues)
        # print("step1 done")
        KalGain, measurementRes = self.KalmanGain(statePredict, covPredict, stateObservation)
        # print("step2 done")
        stateUpdate, covUpdate = self.Update(KalGain, statePredict, measurementRes, covPredict)
        # print("step3 done")
        return stateUpdate, covUpdate



if __name__ == "__main__":

    x = np.array([0,4.5])
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

    a = np.array([np.array([i/20 +random_nums[i][0],i/20 + random_nums[i][1]]) for i in range(1,100)]) # this line adds noise to the exact model 
    actual = np.array([np.array([i/20,i/20]) for i in range(1,100)]) #exact model for plotting purposes

    print(a, "\n")
    prev = np.array([0,0])
    count = 0

    newKalman = KalmanFilter(x,scov,ProcessCov, stateMatrix, controlMatrix, observationMatrix, ObservationCov)
    # todo controlValues, stateObservation, ObservationCov // Will be given inside a time loop
    for i in a:
        # controlValues = np.array([(a[count][0] - prev[0])/deltaT, (a[count][1] - prev[1])/deltaT])
        controlValues = np.array([0.5,0.5])
        newKalman.x, newKalman.prevCov = newKalman.KF(controlValues,i)
        filteredStates.append(newKalman.x)
        filteredCov.append(newKalman.prevCov)
        # prev = i
        prev = newKalman.x
        count = count + 1
        print(newKalman.prevCov)

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
    plt.xlabel('meters')
    plt.ylabel('meters')
    plt.title('Estimation')

    plt.subplot(122)
    plt.plot(error_plotx, label='Error in x')
    plt.plot(error_ploty, label='Error in y')
    plt.xlabel('sample number')
    plt.ylabel('magnitude')
    plt.title('error')


    plt.legend()
    plt.show()