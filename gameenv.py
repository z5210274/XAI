import gym
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box
from gym.envs.registration import register

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

from collections import deque # Ordered collection with ends
###
import pygame, sys
import math
import os.path
import csv
import random

from re import X
from time import time
from pygame.locals import *
from projectile import *

from gametest import Enemy, Player, Projectile

#################################################################################################
#################################################################################################
#################################################################################################

current_time = 0

filename = './data.csv'
FPS = 240

# Screen information
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 900

# Predefined some colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

#################################################################################################
#################################################################################################
#################################################################################################

class GameEnvironment(gym.Env):
    def __init__(self):
        super().__init__() 
        self.enemy_centerx = SCREEN_WIDTH/2
        self.enemy_centery = 80
        self.enemy_theta = 90
        self.player_centerx = SCREEN_WIDTH/2
        self.player_centery = 700
        self.start_pos = [self.enemy_centerx, self.enemy_centery]
        self.current_pos = self.start_pos
        self.mode = 0
        self.reward = 0

        self.P1 = Player()
        self.E1 = Enemy()
        self.projectile_group = pygame.sprite.Group()

        self.action_space = Discrete(6)
        self.observation_shape = (SCREEN_WIDTH,SCREEN_HEIGHT,3)
        self.observation_space = Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        pygame.init()
        self.FramePerSec = pygame.time.Clock()
        self.game_area = pygame.Rect(0,0,720,900)
        self.clock = pygame.time.Clock()

        self.DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.DISPLAYSURF.fill(WHITE)
        pygame.display.set_caption("Game")

    def step(self, action):
        done = False
        if action == 0:  # Right
            self.E1.update(self.enemy_theta, self.mode, action)
        elif action == 1:  # Left
            self.E1.update(self.enemy_theta, self.mode, action)
        #elif action == 2:  # Up
        #    self.E1.update(self.enemy_theta, self.mode, action)
        #elif action == 3:  # Down
        #    self.E1.update(self.enemy_theta, self.mode, action)
        elif action == 2:  # Aim right
            self.E1.update(self.enemy_theta, self.mode, action)
        elif action == 3:  # Aim left
            self.E1.update(self.enemy_theta, self.mode, action)
        elif action == 4:  # Shoot
            self.E1.update(self.enemy_theta, self.mode, action)
        elif action == 5:  # Nothing
            self.E1.update(self.enemy_theta, self.mode, action)

        env.projectile_group.update(self.P1.rect, self.E1, self)
        
        if self.reward > 300:
            done = True
        elif self.reward < 300:
            done = False

        observation = self.get_state()

        return observation, self.reward, done, {}, {}

    def get_state(self):
        arr = pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)
        #arr=arr.dot([0.298, 0.587, 0.114])[:,:,None].repeat(3,axis=2); 
        #state=np.fliplr(np.flip(np.rot90(arr)))
        #state = pygame.surfarray.make_surface(arr)
        return arr

    def render(self, enemy, predict):
        font = pygame.font.SysFont(None,16)
        text = font.render('Aim mode: ' + str(self.E1.aim_text), True, BLACK)
        textRect = text.get_rect()
        textRect.center = (50, 50)

        self.DISPLAYSURF.fill(WHITE)
        self.DISPLAYSURF.blit(text, textRect)
        self.P1.draw(self.DISPLAYSURF)
        self.E1.draw(self.DISPLAYSURF)
        self.projectile_group.draw(self.DISPLAYSURF)
        pygame.draw.line(self.DISPLAYSURF, (238, 75, 43), enemy, predict)

        pygame.display.update()
    
    def reset(self):
        self.enemy_centerx = SCREEN_WIDTH/2
        self.enemy_centery = 80
        self.enemy_theta = 90
        self.player_centerx = SCREEN_WIDTH/2
        self.player_centery = 700
        self.current_pos = self.start_pos
        self.reward = 0

        observation = self.get_state()

        return observation, {}

#################################################################################################
#################################################################################################
#################################################################################################

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_actions = 6

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

def preprocess_frame(frame):
    # Greyscale frame 
    gray = rgb2gray(frame)
    
    # Resize
    preprocessed_frame = transform.resize(gray, [84,84])
    
    return preprocessed_frame # 84x84x1 frame

stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.uint) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.uint) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

#################################################################################################
#################################################################################################
#################################################################################################

register(
    id='Projectile_Predictor',
    entry_point='gameenv:GameEnvironment',
)

env = gym.make('Projectile_Predictor')
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.FrameStack(env, 4)
state = env.reset()
env.render([0,0],[0,0])

while True:
    state = env.reset()
    #state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    env.reward = 0

    for timestep in range (1,max_steps_per_episode):
        aim_x, aim_y = env.E1.aim_calc(env.P1.rect.centerx, env.P1.rect.centery, env.P1.path_history)
        line = [(env.E1.rect.centerx, env.E1.rect.centery),(aim_x, aim_y)]
        theta = getAngle(line[1], line[0])
        if (theta > 360):
            theta = theta - 360

        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        #action = env.action_space.sample()  # Random action selection
        state_next, reward, done, _, hi = env.step(action)
        #state_next, stacked_frames = stack_frames(stacked_frames, state, True)
        state_next = np.array(state_next)

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

         # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            #state_next_sample = preprocess_frame(state_next_sample)
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

        if (action == 4 and env.mode == 0):
            bullet = env.E1.take_shot(env.P1.rect.centerx, env.P1.rect.centery, env.P1.path_history, theta)
            env.projectile_group.add(bullet)

        for event in pygame.event.get():              
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bullet = env.E1.take_shot(env.P1.rect.centerx, env.P1.rect.centery, env.P1.path_history, theta)
                    env.projectile_group.add(bullet)
                if event.key == pygame.K_r: # Clear projectile cache
                    env.projectile_group.empty()
                if event.key == pygame.K_p:
                    env.P1.reset_pos()
                if event.key == pygame.K_o:
                    if (env.P1.mode == 0):
                        env.P1.mode = 1
                    else:
                        env.P1.mode = 0
                if event.key == pygame.K_m:
                    if (env.E1.auto == 0):
                        env.E1.auto = 1
                    else:
                        env.E1.auto = 0
        
        current_time = pygame.time.get_ticks()
        env.P1.update(current_time)
        if (env.E1.auto == 1):
            if (len(env.P1.path_history) > 1000):
                env.E1.strategize(env.P1.rect.centerx, env.P1.rect.centery, env.P1.path_history)
        #env.E1.update(theta, env.mode, 5)
        #env.projectile_group.update(env.P1.rect, env.E1)

        env.render(line[0],line[1])
        env.FramePerSec.tick(FPS)
    
    # Update running reward to check condition for solving
    episode_reward_history.append(env.reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break