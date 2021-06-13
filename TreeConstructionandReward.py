import numpy as np
from matplotlib import pyplot as plt
from math import exp, inf, sqrt

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

def reward_func(child,obs_dist_lat,obs_dist_long,prev_action,prev_reward):
    reward = 0
    k = 1
    if child.pos[0] < 50 and child.pos[0] > 45 and abs(child.pos[1]) < 0.9375:
        return 900
    if child.pos[0] > 50 or abs(child.pos[1]) > 1.875:
        return -inf 
    for i in len(obs_dist_long):
        if obs_dist_long[i][0] < child.pos[0] and obs_dist_long[i][1] > child.pos[0] and obs_dist_lat[i][0] < child.pos[1] and obs_dist_lat[i][1] > child.pos[1]:
            return -inf
        else:
            # TODO reward for distance from obstacles
            pass

    reward = reward + k/(1 + exp(abs(50 - child.pos[0])))
    reward = reward - k*(child.time)
    return reward


def check_validity(node):
    if node.reward == -inf or node.reward == inf:
        return False
    else:
        return True

def generate_nodes(parent,obs_dist_lat,obs_dist_long):
    #0->goto left lane, 1-> goto center, 2->goto right
    # if action_id == 0:
    # one hop
    # create constants for state
    child = parent
    child.pos[0] = child.pos[0] - 0.9375
    child.pos[1] = child.pos[1] + (0.9375*(sqrt(3)))
    child.reward = reward_func(child,obs_dist_lat, obs_dist_long,parent.ego_vel,parent.ego_yaw,parent.prev_action,parent.reward)
    # push in children of the parent 
    parent.children.append(child)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child):
        generate_nodes(parent.children[len(parent.children)-1], 1, obs_dist_lat, obs_dist_long) 

    # if action_id == 1:
    # one hop 
    # create constants for state
    child = parent
    child.pos[0] = child.pos[0] + 0.9375
    child.pos[1] = child.pos[1] + (0.9375*(sqrt(3)))
    child.reward = reward_func(child,obs_dist_lat, obs_dist_long,parent.ego_vel,parent.ego_yaw,parent.prev_action,parent.reward)
    # push in children of the parent 
    parent.children.append(child)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child):
        generate_nodes(parent.children[len(parent.children)-1], 2, obs_dist_lat, obs_dist_long)
    # if action_id == 2:
        # one hop
        # pass
    child = parent
    child.pos[0] = child.pos[0] + 0.9375
    child.pos[1] = child.pos[1] + (0.9375*(sqrt(3)))
    child.reward = reward_func(child,obs_dist_lat, obs_dist_long,parent.ego_vel,parent.ego_yaw,parent.prev_action,parent.reward)
    # push in children of the parent 
    parent.children.append(child)
    # call generate node on this generated node and pass parent as, parent.children[length(children)-1]
    if check_validity(child):
        generate_nodes(parent.children[len(parent.children)-1], 3, obs_dist_lat, obs_dist_long)
    return 

def DFS(parent, node_list):
    if parent.reward == -inf:
        node_list.append(parent)
        return node_list
    if parent.reward == 900:
        node_list.append(parent)
        return node_list
    
    for i in range(len(parent.children)):
        node_list.append(DFS(parent.children[i],node_list))
    
    return node_list
    
