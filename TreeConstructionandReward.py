import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt
import copy 
import time
import sys

terminatingRew = -inf
goalRew = 50


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
    k = 10
    if child.pos[0] >= scene_len or abs(child.pos[1]) >= 1.875:
        return terminatingRew
    if child.time >= scene_len/6.33:
        return terminatingRew
    for i in range(len(obs_dist_long)):
        if obs_dist_long[i][0] < child.pos[0] and obs_dist_long[i][1] > child.pos[0] and obs_dist_lat[i][0] < child.pos[1] and obs_dist_lat[i][1] > child.pos[1]:
                return terminatingRew
        else:
            reward = reward -1
    if child.pos[0] < scene_len and child.pos[0] > scene_len-1 and abs(child.pos[1]) < 0.9375 :
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

def dynamic_obs_gen(deltaT,u,a,long_ini,lat_ini):
    # long_ini = [[10,12.5]]
    # lat_ini = [[-1.875,-0.9375]]

    # add a for loop for velocity and accelaration of vehicles here if you want multiple obstacles
    for iter in range(len(u)):
        for i in long_ini:
            # updating longitudinal coordinates
            i[0] = (u[iter]*deltaT) + 0.5*a[iter]*(deltaT**2) + i[0]
            i[1] = (u[iter]*deltaT) + 0.5*a[iter]*(deltaT**2) + i[1]
    return long_ini, lat_ini



def generate_nodes(parent,obs_lat_ini,obs_long_ini,scene_len,obs_vel, obs_acc):
    #0->go left lane, 1-> stay, 2->go right


    # generating position of dynamic obstacles given the previous position and simple mathematical model of the obstacles.
    obs_dist_long, obs_dist_lat = dynamic_obs_gen(1.875/6.33,obs_vel, obs_acc, obs_long_ini, obs_lat_ini)

    child = state(0,6.33,0,0,0,0,0,0,0)
    child1 = state(0,6.33,0,0,0,0,0,0,0)
    child2 = state(0,6.33,0,0,0,0,0,0,0)
    
    child1.pos[0] = parent.pos[0] + 1.875
    child1.pos[1] = parent.pos[1] + 0
    child1.action = 1
    child1.time =  parent.time + (1.875/6.33)
    child1.reward = reward_func(child1,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)

    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child1):
        child1 = generate_nodes(child1, obs_dist_lat, obs_dist_long,scene_len,obs_vel, obs_acc)

    # push in children of the parent 
    parent.children.append(child1)

    # print(parent.pos[1])  
    child.pos[0] = parent.pos[0] + (0.9375*(sqrt(3)))
    child.pos[1] = parent.pos[1] - 0.9375
    child.action = 0
    child.reward = reward_func(child,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)
    # child.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
    child.time = parent.time + (1.875/6.33)
    # push in children of the parent 
    
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child):
        # print(child.pos[0]) 
        child = generate_nodes(child, obs_dist_lat, obs_dist_long,scene_len,obs_vel, obs_acc) 
    parent.children.append(child)


    child2.pos[0] = parent.pos[0] + 0.9375*(sqrt(3))
    child2.pos[1] = parent.pos[1] + 0.9375
    child2.action = 2
    child2.reward = reward_func(child2,obs_dist_lat, obs_dist_long,parent.action,parent.reward,scene_len)
    # child2.time =  parent.time + (sqrt(0.935**2 + (0.935*sqrt(3))**2)/6.33)
    child2.time = parent.time + (1.875/6.33)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child2):
        # print(child2.pos[0])
        child2 = generate_nodes(child2, obs_dist_lat, obs_dist_long,scene_len,obs_vel, obs_acc)
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
    obs_vel = [6.33]
    obs_acc = [0]
    scene_len = 20
    # for count in scene_len:
    t = time.time()
    temp = generate_nodes(temp, obs_lat, obs_long,scene_len,obs_vel, obs_acc)
    print("Time taken to generate tree", time.time() - t)
    ans = []
    ans1 = [[]]
    t1 = time.time()
    dt,_ = DFS(temp)
    print("Time taken to traverse tree", time.time() - t1)
    bestBranch(dt)
