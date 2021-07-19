import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
import time
import TreeConstructionandReward as tree
from TreeConstructionandReward import KalmanFilter
import gc

def findEgoVehLoc(dt,t):
    for i in dt:
        if i.time > t:
            return i
    return dt[len(dt)-1]

def ObservationGenerator(prevRolloutTime, currT, prevObservation):
    vehicle_velocity = np.array([0,1])
    prevObservation = prevObservation + vehicle_velocity.dot(currT-prevRolloutTime)
    return prevObservation

def updatePath(ego_pos, current_path):
    for i in current_path:
        i.pos[1] = i.pos[1] - ego_pos.pos[1]
    return current_path


def initilisation(obstacleInitialState, obstacleVelocity):
    temp = tree.state(0,6.33,0,0,0,0,0,0,time.time()/10000000000)
    obs_vel = obstacleVelocity
    scene_len = 15

    ############################ KALMAN FILTER INITIALISATION ###########################
    x = obstacleInitialState
    deltaT = 1.875/6.33
    scov = np.eye(2) # initial value of Process covariance matrix 
    scov[0][0] = 0.04
    scov[0][1] = 0.0
    scov[1][0] = 0.0 
    scov[1][1] = 0.1


    stateMatrix = np.eye(2) # Ak
    controlMatrix = np.array([[deltaT,0],[0,deltaT]]) #Bk
    observationMatrix = np.eye(2) #Hk
    ObservationCov = np.array([[0.5,0.4],[0.4,0.9]]) #(R_k)
    ProcessCov = np.array([[0,0],[0,0]]) # process covariance matrix value (Q_k)

    newKalman = KalmanFilter(x,scov,ProcessCov, stateMatrix, controlMatrix, observationMatrix, ObservationCov)

    ####################################################################################

    obstacle_info = tree.obstacle(x[0],x[1],scov,obs_vel[0],obs_vel[1],0,0)

    return temp, newKalman, obstacle_info, scene_len


if __name__ == "__main__":
    obs = ObservationGenerator(0,0,np.array([0,5.75]))
    # initialise obstacle variable and other scene info variables
    temp, newKalman, obstacle_info, scene_len = initilisation(obs, [0,1])
    prevT = 0
    t = time.time()
    temp = tree.generate_nodes(temp, obstacle_info,scene_len,newKalman)
    ans1 = [[]]
    dt,_ = tree.DFS(temp)
    current_path = tree.bestBranch(dt)
    ans1.clear()
    t = time.time() - t
    prevRolloutTime = t
    while(True):
        # update ego vehicle position 
        ego_pos = findEgoVehLoc(current_path,t)
        if(t - prevRolloutTime >= 1):
            temp = tree.state(0,6.33,0,0,0,0,0,0,time.time()/10000000000)
            current_path = updatePath(ego_pos, current_path)
            print("New Sensor data arrived; tree start")
            # Call obstacle generator -- and reinitialise in the local coordinate frame(0-scene-len); put car in 0 again
            timestop = time.time()
            obs = ObservationGenerator(prevRolloutTime, t, obs)
            newKalman.x, newKalman.prevCov = newKalman.KF(np.array([obstacle_info.vel_lat,obstacle_info.vel_long]),obs)
            obstacle_info.cov = newKalman.prevCov
            obstacle_info.mean = newKalman.x
            # call tree function 
            deltatime = time.time()
            print(deltatime-timestop)
            temp = tree.generate_nodes(temp, obstacle_info,scene_len,newKalman)
            print(time.time() - deltatime, "Tree node generation time")
            ans1 = [[]]
            dt,_ = tree.DFS(temp)
            ans1.clear()
            current_path = tree.bestBranch(dt)
            #TODO DEBUG
            # for k in current_path:
            #     # print([k.pos[0],k.pos[1],k.time])   
            #     # print(obs_long)
            #     pass
            deltatime = time.time() - deltatime
            prevRolloutTime = t
            prevT = t
            t  = t + deltatime
            print(t)
            del temp
            gc.collect()
            continue
        print(t)
        prevT = t
        t = t + 0.1