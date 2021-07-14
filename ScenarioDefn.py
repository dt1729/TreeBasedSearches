import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
import time
import TreeConstructionandReward as tree
from TreeConstructionandReward import KalmanFilter

def findEgoVehLoc(dt,t):
    for i in dt:
        if i.time > t:
            return i
    return dt[len(dt)-1]

if __name__ == "__main__":
    # initialise obstacle variable and other scene info variables
    obs_lat = [[-1.875,-0.9375]]
    obs_long = [[10,12.5]]
    obs_vel = [6.33]
    obs_acc = [0]
    scene_len = 20
    temp = tree.state(0,30,0,0,0,0,0,0,0)
    prevT = 0
    t = time.time()
    temp = tree.generate_nodes(temp, obs_lat, obs_long,scene_len,obs_vel, obs_acc)
    ans1 = [[]]
    dt,_ = tree.DFS(temp,ans1)
    current_path = tree.bestBranch(dt)
    ans1.clear()
    t = time.time() - t
    prevRolloutTime = t
    while(True):
        # update ego vehicle position 
        ego_pos = findEgoVehLoc(current_path,t)
        # update obstacle variables
        obs_long, obs_lat = tree.dynamic_obs_gen(t - prevT,obs_vel,obs_acc,obs_lat,obs_long)
        if(t - prevRolloutTime >= 1):
            print("New Sensor data arrived; tree start")
            for i in obs_long:
                for k in i:
                    k = k - ego_pos.pos[0]
            # call tree function 
            deltatime = time.time()
            temp = tree.generate_nodes(temp, obs_lat, obs_long,scene_len,obs_vel, obs_acc)
            ans1 = [[]]
            dt,_ = tree.DFS(temp,ans1)
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
            continue
        print(t)
        prevT = t
        t = t + 0.1