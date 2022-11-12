import os
import gym
import minerl
import logging
logging.basicConfig(level=logging.DEBUG)
from time import sleep
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
import time
import math
env = gym.make('MineRLBasaltFindCave-v0')

# Note that this command will launch the MineRL environment, which takes time.
# Be patient!
obs = env.reset()

done = False
score = 0
frame = 0
print(env.action_space)

# internal compass
starting_location = [0,0] # x, z   | no y to keep it simple
current_position = [0,0]
history = []
history.append(starting_location)
heading = 0

WALKING_SPEED = 4.317 # 4.317 blocks per second | 20 steps per second
SPRINTING_SPEED = 5.612 # 5.612 blocks per second | 20 steps per second
WALKING_SPEED_PER_STEP = WALKING_SPEED/20
SPRINTING_SPEED_PER_STEP = SPRINTING_SPEED/20
distance_from_last_point = 0
# distance_walked = WALKING_SPEED_PER_STEP * (whatever step)

def camera_mov(heading, mov):
    if(heading + mov < 0):
        heading = 360 - abs(heading + mov)
    elif(heading + mov > 360):
        heading = abs(heading + mov) - 360
    else:
        heading += mov
    return heading

def coordinates(heading, history, action, distance_from_last_point, current_position):
    if(action["sprint"] == 1):
        z = (math.sin(math.radians(heading)) * SPRINTING_SPEED_PER_STEP)
        x = (math.cos(math.radians(heading)) * SPRINTING_SPEED_PER_STEP)
    else:
        z = (math.sin(math.radians(heading)) * WALKING_SPEED_PER_STEP)
        x = (math.cos(math.radians(heading)) * WALKING_SPEED_PER_STEP)

    blocks_from_point = 10
    
    if(action['forward'] == 1):
        print("Forward")
        current_position[0] += x
        current_position[1] += z
    elif(action['left'] == 1):
        current_position[0] += -z
        current_position[1] += x       
    elif(action['right'] == 1):
        current_position[0] += z
        current_position[1] += x        
    elif(action['back'] == 1):    
        current_position[0] += x
        current_position[1] += -z         

    distance_from_last_point = math.sqrt((current_position[0] - history[-1][0])**2 + (current_position[1] - history[-1][1])**2)
    if(distance_from_last_point > blocks_from_point):
        temp_pos =[0,0]
        temp_pos[0] = current_position[0]
        temp_pos[1] = current_position[1]
        history.append(temp_pos) # update
        distance_from_last_point = 0 # reset
        print("Updating....")
    return heading, history, distance_from_last_point, current_position

def too_close(current_position, history):
    for i in history:
        if(((current_position[0] - history[i][0])**2 + (current_position[1] - history[i][1])**2) <= 5**2):
            reward = -3
            break

    return reward

def remove_actions(action):
    action["ESC"] = 0
    action["inventory"] = 0 
    action["drop"] = 0
    action["use"] = 0
    action["swapHands"] = 0
    action["sneak"] = 0 
    action["hotbar.1"] = 0
    action["hotbar.2"] = 0
    action["hotbar.3"] = 0
    action["hotbar.4"] = 0
    action["hotbar.5"] = 0
    action["hotbar.6"] = 0
    action["hotbar.7"] = 0
    action["hotbar.8"] = 0
    action["hotbar.9"] = 0

    # action["camera"] = [0,0]

    # action["forward"] = 1

action = env.action_space.sample()
remove_actions(action)
done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    # action["inventory"] = 0
    remove_actions(action)

    if(action['camera'][1] != 0):
        n_h = camera_mov(heading, action['camera'][1])
        heading = n_h

    # # can you press forward and left at the same time??
    if((action['forward'] == 1) or (action['left']==1) or (action['right']==1) or (action['back']==1)):
        heading, history, distance_from_last_point, current_position = coordinates(heading, history, action, distance_from_last_point, current_position)
        print("history: ", history, "   | current_position: ", current_position)

    if((action['forward'] == 1) and (action['left']==1) or ((action['forward']==1) and (action['right']==1))):
        print("###############   Multiple buttons pressed ###############")

    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
