import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
import time
import TreeConstructionandReward as tree

def findEgoVehLoc(dt,t):
    for i in dt:
        if i[2] > t:
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

    t = time.time()
    temp = tree.generate_nodes(temp, obs_lat, obs_long,scene_len,obs_vel, obs_acc)
    dt,_ = tree.DFS(temp)
    current_path = bestBranch(dt)
    t = time.time() - t

    while(True):
        # update ego vehicle position 
        ego_pos = findEgoVehLoc(dt,t)
        # update obstacle variables
        obs_long, obs_lat = tree.dynamic_obs_gen(t - prevT,obs_vel[0],obs_acc[0],obs_lat,obs_long)

        if(time.time() - t >= 1):
            for i in obs_long:
                i[0] = i[0] - ego_pos[0]
                i[1] = i[1] - ego_pos[0]
            # call tree function 
            deltatime = time.time()
            temp = tree.generate_nodes(temp, obs_lat, obs_long,scene_len,obs_vel, obs_acc)
            dt,_ = tree.DFS(temp)
            current_path = bestBranch(dt)
            deltatime = time.time() - deltatime
            prevT = t
            t  = t + deltatime
            continue

        prevT = t
        t = t + 0.1