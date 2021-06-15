import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
import copy 
import time
import sys
class state:
    def __init__(self, reward, ego_vel, ego_yaw, prev_action, prev_reward,longitudinal,lane_offset,action,time) -> None:
        # here time  = prev node time + action time.
        self.reward = reward
        self.ego_vel = ego_vel
        self.ego_yaw = ego_yaw
        self.children = []
        self.pos = [longitudinal,lane_offset]
        self.action = action
        self.time = time


def add_node(parent, child):
    parent.children.append(child)

def reward_func(child,obs_dist_lat,obs_dist_long,prev_action,prev_reward,scene_len):
    reward = 0
    k = 1
    if child.pos[0] < scene_len and child.pos[0] > scene_len-1 and abs(child.pos[1]) < 0.9:
        return 900
    if child.pos[0] >= scene_len or abs(child.pos[1]) >= 1.875:
        return -inf 
    if child.time > scene_len/6.33:
        return -inf
    # for i in range(len(obs_dist_long)):
    #     if obs_dist_long[i][0] < child.pos[0] and obs_dist_long[i][1] > child.pos[0] and obs_dist_lat[i][0] < child.pos[1] and obs_dist_lat[i][1] > child.pos[1]:
    #           return -inf
    #     else:
    #         # TODO reward for distance from obstacles 
    #         pass

    reward = reward + k/(1 + exp(abs(20 - child.pos[0])))
    reward = reward - k*(child.time)
    return reward


def check_validity(node):
    if node.reward == -inf or node.reward == 900:
        return False
    else:
        return True

def generate_nodes(parent,obs_dist_lat,obs_dist_long,scene_len):
    #0->go left lane, 1-> stay, 2->go right
    if parent.pos[0] > scene_len and abs(parent.pos[1]) >= 1.875:
        return parent

    child = state(0,30,0,0,0,0,0,0,0)
    child1 = state(0,30,0,0,0,0,0,0,0)
    child2 = state(0,30,0,0,0,0,0,0,0)
    
    child1.pos[0] = parent.pos[0] + 1.875
    child1.pos[1] = parent.pos[1] + 0
    child1.action = 1
    child1.time =  parent.time + (1.875/6.33)
    child1.reward = reward_func(child1,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)
    # push in children of the parent 

    # if(child1.time > (10/6.33)):
    #     print(child1.time)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child1):
        child1 = generate_nodes(child1, obs_dist_lat, obs_dist_long,scene_len)
    parent.children.append(child1)

    # print(parent.pos[1])  
    child.pos[0] = parent.pos[0] + (0.9375*(sqrt(3)))
    child.pos[1] = parent.pos[1] - 0.9375
    child.action = 0
    child.reward = reward_func(child,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)
    child.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
    # push in children of the parent 
    
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child):
        # print(child.pos[0]) 
        child = generate_nodes(child, obs_dist_lat, obs_dist_long,scene_len) 
    parent.children.append(child)


    child2.pos[0] = parent.pos[0] + 0.9375*(sqrt(3))
    child2.pos[1] = parent.pos[1] + (0.9375)
    child2.action = 2
    child2.reward = reward_func(child2,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)
    child2.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
    # push in children of the parent 
    # if(child2.time > (10/6.33)):
    #     print(child1.time)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child2):
        # print(child2.pos[0])
        child2 = generate_nodes(child2, obs_dist_lat, obs_dist_long,scene_len)
    parent.children.append(child2)
    return parent

def merge(a,b):
    for j in b:
        a.append(j)
    return a

def DFS(parent, node_list,list_list):
    if parent.reward == -inf:
        temp = list()  
        temp.append(parent)
        return temp,False
    if parent.reward == 900:
        temp = list()  
        temp.append([parent])
        return temp,True

    a = False
    b = []

    for i in range(len(parent.children)):
        temp_store,bol = DFS(parent.children[i],node_list, b)
        a = a or bol
        if bol == True:
            for t in temp_store:
                try:
                    t.append(parent)
                except:
                    print(t.reward)

            b = merge(b,temp_store)
    return b,a    

def BFS(parent):
    pass
     
if __name__ == "__main__":
    temp = state(0,30,0,0,0,0,0,0,0)
    obs_lat = [[-1.975,-2.475]]
    obs_long = [[10,10.5]]
    scene_len = [20,22,24,25,26,30]
    for count in scene_len:
        t= time.time()
        temp = generate_nodes(temp, obs_lat, obs_long,count)
        print("Time taken to generate tree", time.time() - t)
        ans = []
        ans1 = [[]]
        t1 = time.time()
        dt,_ = DFS(temp,ans,ans1)
        print("Time taken to traverse tree", time.time() - t1)
        # break
    # for i in dt:
    #     # print(len(i))
    #     for j in i:
    #         print(j.pos[0], j.pos[1], j.reward)  
    #     print("\n")