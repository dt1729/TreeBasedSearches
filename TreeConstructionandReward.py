import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
from numpy.random import default_rng
import copy 
import time
import sys

terminatingRew = -inf
goalRew = 50

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
        # print(stateEstimate)
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

class state:
    def __init__(self, reward, ego_vel, ego_yaw, prev_action, prev_reward,longitudinal,lane_offset,action,time) -> None:
        # here time  = prev node time + action time.
        self.reward = reward
        self.ego_vel = ego_vel
        self.ego_yaw = ego_yaw
        self.children = []
        self.pos = [lane_offset,longitudinal]
        self.action = action
        self.time = time

class obstacle:
    def __init__(self, obs_mean_ini_lat, obs_mean_ini_long, obs_cov_ini,obs_vel_ini_lat,obs_vel_ini_long, obs_acc_ini_lat,obs_acc_ini_long):
        # Expecting these to be single values except covariance, generates particles itself
        self.cov = obs_cov_ini
        self.mean = [obs_mean_ini_lat,obs_mean_ini_long]
        self.vel_lat = obs_vel_ini_lat
        self.vel_long = obs_vel_ini_long
        self.acc_lat = obs_acc_ini_lat
        self.acc_long = obs_acc_ini_long

    # extend this for multiple obstacles
    def generate_sample(self):
        coords = np.random.multivariate_normal(self.mean, self.cov).T #x,y are lat and long
        # print(self.cov)
        return coords #return in the form of [lat, long]


def add_node(parent, child):
    parent.children.append(child)

#TODO add obstacle class object
def reward_func(child,obs_dist_lat,obs_dist_long,prev_action,prev_reward,scene_len):
    reward = 0
    k = 10
    if child.pos[0] >= scene_len or abs(child.pos[1]) >= 1.875:
        return terminatingRew
    if child.time >= scene_len/6.33:
        return terminatingRew
    # for i in range(len(obs_dist_long)):
    if obs_dist_long-2.35 < child.pos[0] and obs_dist_long+2.35 > child.pos[0] and obs_dist_lat-1.45 < child.pos[1] and obs_dist_lat+1.45 > child.pos[1]:
            # print(child.pos, obs_dist_lat, obs_dist_long)
            return terminatingRew
    else:
        reward = reward -1
    if child.pos[0] < scene_len and child.pos[0] > scene_len-2 and abs(child.pos[1]) < 1.875 :
        return goalRew

    # reward = reward + 100/(1 + exp(abs(scene_len - child.pos[0])))
    # reward = reward - k*abs(scene_len - child.pos[0])/scene_len
    # reward = reward - k*abs(0 - child.pos[1])/1.875
    # reward = reward - k*abs((scene_len/6.33) - child.time)/(scene_len/6.33)
    reward = reward - k*(abs(child.pos[1]))/1.875
    reward = reward - 1
    return reward


def check_validity(node):
    if node.reward == terminatingRew or node.reward == goalRew:
        return False
    else:
        return True

# TODO obstacle class object to be passed
def dynamic_obs_gen(deltaT,u,a,long_ini,lat_ini):
    # add a for loop for velocity and accelaration of vehicles here if you want multiple obstacles
    for iter in range(len(u)):
        for i in long_ini:
            # updating longitudinal coordinates
            i[0] = (u[iter]*deltaT) + 0.5*a[iter]*(deltaT**2) + i[0]
            i[1] = (u[iter]*deltaT) + 0.5*a[iter]*(deltaT**2) + i[1]
    return long_ini, lat_ini

def dynamic_obs_stochastic(deltaT,obstacle):
    for i in obstacle.particles:
            i[0] = (obstacle.vel_lat*deltaT) + 0.5*(obstacle.acc_lat)*(deltaT**2) + i[0] + np.random.normal(mean,std,1)
            i[1] = (obstacle.vel_long*deltaT) + 0.5*(obstacle.acc_long)*(deltaT**2) + i[1] + np.random.normal(mean,std,1)
    return obstacle


def generate_nodes(parent,obstacle_info,scene_len,newKalman):
    #0->go left lane, 1-> stay, 2->go right
    obs_info = []
    newKalman.x, newKalman.prevCov = newKalman.Predict([obstacle_info.vel_lat,obstacle_info.vel_long])
    obstacle_info.cov = newKalman.prevCov
    obstacle_info.mean = newKalman.x
    # print(newKalman.x)
    for i in range(2):
        obs_info = obstacle_info.generate_sample()
        print(obstacle_info.cov)
        obs = obstacle(obs_info[0],obs_info[1], obstacle_info.cov,obstacle_info.vel_lat,obstacle_info.vel_long,0,0)
        # generating position of dynamic obstacles given the previous position and simple mathematical model of the obstacles.

        child = state(0,6.33,0,0,0,0,0,0,0)
        child1 = state(0,6.33,0,0,0,0,0,0,0)
        child2 = state(0,6.33,0,0,0,0,0,0,0)
        
        child1.pos[0] = parent.pos[0] + 1.875
        child1.pos[1] = parent.pos[1] + 0
        child1.action = 1
        child1.time =  parent.time + (1.875/6.33)
        child1.reward = reward_func(child1,obs.mean[0], obs.mean[1],parent.action,parent.reward,scene_len)

        # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
        if check_validity(child1):
            child1 = generate_nodes(child1, obs ,scene_len,newKalman)

        # push in children of the parent 
        parent.children.append(child1)

        # print(parent.pos[1])  
        child.pos[0] = parent.pos[0] + (0.9375*(sqrt(3)))
        child.pos[1] = parent.pos[1] - 0.9375
        child.action = 0
        child.reward = reward_func(child,obs.mean[0], obs.mean[1],parent.action,parent.reward,scene_len)
        # child.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
        child.time = parent.time + (1.875/6.33)
        # push in children of the parent 
        
        # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
        if check_validity(child):
            # print(child.pos[0]) 
            child = generate_nodes(child, obs ,scene_len,newKalman) 
        parent.children.append(child)


        child2.pos[0] = parent.pos[0] + 0.9375*(sqrt(3))
        child2.pos[1] = parent.pos[1] + 0.9375
        child2.action = 2
        child2.reward = reward_func(child2,obs.mean[0], obs.mean[1],parent.action,parent.reward,scene_len)
        # child2.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
        child2.time = parent.time + (1.875/6.33)
        # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
        if check_validity(child2):
            # print(child2.pos[0])
            child2 = generate_nodes(child2, obs ,scene_len,newKalman)
        parent.children.append(child2)
    return parent

def merge(a,b):
    for j in b:
        a.append(j)
    return a

def DFS(parent):
    if parent.reward == terminatingRew:
        temp = list()  
        temp.append(parent)
        return temp,False
    if parent.reward == goalRew:
        temp = list()  
        temp.append([parent])
        return temp,True

    a = False
    b = []

    for i in range(len(parent.children)):
        temp_store,bol = DFS(parent.children[i])
        a = a or bol
        if bol == True:
            for t in temp_store:
                try:
                    t.append(parent)
                except:
                    print(t.reward)
            b = merge(b,temp_store)
    return b,a    

def bestBranch(dt):
    prev_sum = -inf
    sum = 0
    best_branch = []
    for i in dt:
        for k in i:
            sum = sum + k.reward
        if sum > prev_sum:
            # print(sum,"\n")
            print([[i[p].pos[0],i[p].pos[1], i[p].time] for p in range(len(i))],"\n")
            print([[i[k-1].action,i[k].pos[1]] for k in range(len(i))],"\n")
            ansPos = [[i[p].pos[0],i[p].pos[1], i[p].time] for p in range(len(i))]
            ansAction = [[i[p-1].action,i[p].pos[1]] for p in range(len(i))]
            # please remember that you've mapped actions incorrectly
            prev_sum = sum
            best_branch = i
        sum = 0
    return best_branch

def BFS(parent):
    pass
     
if __name__ == "__main__":
    temp = state(0,6.33,0,0,0,0,0,0,time.time()/10000000000)
    obs_lat = [[-0.875,0.875]]
    obs_long = [[10,11.5]]
    obs_vel = [0,0]
    obs_acc = [0,0]
    scene_len = 15

    
    # for count in scene_len:
    ############################ KALMAN FILTER INITIALISATION ###########################
    x = np.array([0,10.75])
    deltaT = 1.875/6.33
    scov = np.eye(2) # initial value of Process covariance matrix 
    scov[0][0] = 0.1
    scov[0][1] = 0.0
    scov[1][0] = 0.0 
    scov[1][1] = 0.1

    stateMatrix = np.eye(2) # Ak
    controlMatrix = np.array([[deltaT,0],[0,deltaT]]) #Bk
    observationMatrix = np.eye(2) #Hk
    ObservationCov = np.array([[0.5,0.4],[0.4,0.9]]) # (R_k)
    ProcessCov = np.array([[0,0],[0,0]]) # process covariance matrix value (Q_k)

    newKalman = KalmanFilter(x,scov,ProcessCov, stateMatrix, controlMatrix, observationMatrix, ObservationCov)

    ####################################################################################

    obstacle_info = obstacle(0,10.75,scov,obs_vel[0],obs_vel[1],0,0)

    t = time.time()
    temp = generate_nodes(temp, obstacle_info,scene_len,newKalman)
    print("Time taken to generate tree", time.time() - t)
    ans = []
    ans1 = [[]]
    t1 = time.time()
    dt,_ = DFS(temp)
    print("Time taken to traverse tree", time.time() - t1)
    bestBranch(dt)
