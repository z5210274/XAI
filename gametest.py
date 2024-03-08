import pygame, sys
import math
import os.path
import csv
import random
import numpy as np

from re import X
from time import time
from pygame.locals import *
from projectile import *
from movement import *

current_time = 0

filename = './data.csv'
FPS = 240

# Screen information
#SCREEN_WIDTH = 720
#SCREEN_HEIGHT = 900
SCREEN_WIDTH = 360
SCREEN_HEIGHT = 450

# Predefined some colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

check_file = os.path.isfile(filename)
print("data.csv exists: " + str(check_file))

def write_csv(new_data):
    field_names = ["Shooter_x_pos","Shooter_y_pos",
                "Projectile_x_pos","Projectile_y_pos",
                "Player_x_pos_current","Player_y_pos_current",
                "Player_x_pos_initial","Player_y_pos_initial",
                "Theta",
                #"Blocked",
                "Hit"]
    dict = new_data
    with open(filename, 'a') as file:
        dict_object = csv.DictWriter(file, fieldnames=field_names, lineterminator = '\n') 
        dict_object.writerow(dict)

#################################################################################################
#################################################################################################
#################################################################################################
 
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Enemy.png"), (25,25))
        #self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect() 
        self.rect.center = (SCREEN_WIDTH/2, 80)
        self.x = self.rect.centerx 
        self.y = self.rect.centery
        self.theta = 90
        self.projectile_range = 800
        self.sensitivity = 1
        self.move_speed = 1
        self.aim_mode = 3 # 0 - Neuro, 1 - Social, 2 - Cultural, 3 - Player
        self.aim_text = 'Player'
        self.auto = 0
        self.reloading = 0

        self.neuro = 0.75
        self.social = 0.75
        self.cultural = 0.5
        self.learning_rate = 0.05

    def reset(self):
        self.rect.center = (SCREEN_WIDTH/2, 80)
        self.x = self.rect.centerx 
        self.y = self.rect.centery
        self.theta = 90
 
    def update(self, theta, mode, action):
        if mode == 1:
            pressed_keys = pygame.key.get_pressed()

            if pressed_keys[K_1]:
                self.aim_mode = 0
                self.aim_text = 'Neuro'
            if pressed_keys[K_2]:
                self.aim_mode = 1
                self.aim_text = 'Social'
            if pressed_keys[K_3]:
                self.aim_mode = 2
                self.aim_text = 'Cultural'
            if pressed_keys[K_4]:
                self.aim_mode = 3
                self.aim_text = 'Player'

            if self.rect.left > 0:
                if pressed_keys[K_a]:
                    self.rect.move_ip(-self.move_speed, 0)
            if self.rect.right < SCREEN_WIDTH:        
                if pressed_keys[K_d]:
                    self.rect.move_ip(self.move_speed, 0)
            if self.rect.top > 0:
                if pressed_keys[K_w]:
                    self.rect.move_ip(0, -self.move_speed)
            if self.rect.bottom < SCREEN_HEIGHT:        
                if pressed_keys[K_s]:
                    self.rect.move_ip(0, self.move_speed)

            if (self.aim_mode == 0):
                self.update_neuro_mode()
                self.theta = theta
            if (self.aim_mode == 1):
                self.update_social_mode()
                self.theta = theta
            if (self.aim_mode == 2):
                self.update_cultural_mode()
                self.theta = theta
            if (self.aim_mode == 3):
                self.update_player_mode()

        if mode == 0:
            if self.rect.left > 0:
                if action == 0:
                    self.rect.move_ip(-self.move_speed, 0)
            if self.rect.right < SCREEN_WIDTH:        
                if action == 1:
                    self.rect.move_ip(self.move_speed, 0)
            '''if self.rect.top > 0:
                if action == 2:
                    self.rect.move_ip(0, self.move_speed)
            if self.rect.bottom < SCREEN_HEIGHT:        
                if action == 3:
                    self.rect.move_ip(0, -self.move_speed)'''
            if action == 2:
                self.theta += -self.sensitivity
            if action == 3:
                self.theta += self.sensitivity
                
                
            if (self.theta > 360):
                self.theta += -360
            if (self.theta < 0):
                self.theta += 360
    
    def update_neuro_mode(self):
        return 0

    def update_social_mode(self):
        return 0

    def update_cultural_mode(self):
        return 0

    def update_player_mode(self):
        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[K_LEFT]:
            self.theta += self.sensitivity
        if pressed_keys[K_RIGHT]:
            self.theta += -self.sensitivity
        
        if (self.theta > 360):
            self.theta += -360
        if (self.theta < 0):
            self.theta += 360

        self.x = self.rect.centerx
        self.y = self.rect.centery

    def strategize(self, player_x, player_y, path_history):
        neuro_x = path_history[0][1] - player_x
        neuro_y = path_history[0][2] - player_y
        dist_neuro = math.sqrt(neuro_x**2 + neuro_y**2)
        neuro_weight = self.neuro*dist_neuro

        social_x = path_history[800][1] - player_x
        social_y = path_history[800][2] - player_y
        dist_social = math.sqrt(social_x**2 + social_y**2)
        social_weight = self.social*dist_social

        cultural_x = path_history[600][1] - player_x
        cultural_y = path_history[600][2] - player_y
        dist_cultural = math.sqrt(cultural_x**2 + cultural_y**2)
        cultural_weight = self.cultural*dist_cultural

        #print(neuro_weight, social_weight, cultural_weight)

        if (neuro_weight >= max(social_weight, cultural_weight)):
            self.aim_mode = 0
            self.aim_text = 'Neuro'
        elif (social_weight >= cultural_weight):
            self.aim_mode = 1
            self.aim_text = 'Social'
        else:
            self.aim_mode = 2
            self.aim_text = 'Cultural'


    def take_shot(self, player_x, player_y, path_history, theta):

        bullet = Projectile(self.rect.centerx, self.rect.centery, player_x, player_y, theta, self.aim_mode)

        return bullet

    def aim_calc(self, player_x, player_y, path_history):
        if (self.aim_mode == 0):
            x = player_x
            y = player_y
        if (self.aim_mode == 1):
            dist = math.sqrt((player_x - self.x)**2 + (player_y - self.x)**2)
            time_estimate = int(dist/3.5)
            prev = path_history[(len(path_history)-1) - time_estimate]
            x_move = player_x - prev[1]
            y_move = player_y - prev[2]
            x = player_x + x_move
            y = player_y + y_move
        if (self.aim_mode == 2):
            dist = math.sqrt((player_x - self.x)**2 + (player_y - self.x)**2)
            time_estimate = int(dist/7)
            prev = path_history[(len(path_history)-1) - time_estimate]
            x_move = player_x - prev[1]
            y_move = player_y - prev[2]
            x = player_x - x_move
            y = player_y - y_move
        if (self.aim_mode == 3):
            theta = toRadian(self.theta)
            x = self.x + self.projectile_range*math.cos(theta)
            y = self.y + self.projectile_range*math.sin(theta)
        
        return (x,y)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 
    
#################################################################################################
#################################################################################################
#################################################################################################
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Player.png"), (25,25))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT - 100)
        self.dest_x = -1
        self.dest_y = -1
        self.x = self.rect.centerx
        self.y = self.rect.centery
        self.x_vel = 0
        self.y_vel = 0
        self.theta = 0
        self.move_speed = 1
        self.path_history = []
        self.mode = 0
        self.move_right = 1
        self.juke = -1
        self.dist_travelled = 0
        self.boosted = -1

    def reset(self):
        self.rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT - 100)
        self.x = self.rect.centerx
        self.y = self.rect.centery
 
    def update(self, current_time, collectable_group):
        self.update_path_history(current_time)
        
        if (self.boosted >= 0):
            self.boosted += 1

            if (self.boosted >= 5000):
                self.boosted = -1
                self.move_speed = self.move_speed/1.5
        
        if (self.mode == 0):
            movement(sys.argv[1], SCREEN_WIDTH, SCREEN_HEIGHT, self, collectable_group)

        if (self.mode == 1):
            pressed_keys = pygame.key.get_pressed()
         
            if self.rect.left > 0:
                if pressed_keys[K_j]:
                    self.rect.move_ip(-self.move_speed, 0)
            if self.rect.right < SCREEN_WIDTH:        
                if pressed_keys[K_l]:
                    self.rect.move_ip(self.move_speed, 0)
            if self.rect.top > 0:
                if pressed_keys[K_i]:
                    self.rect.move_ip(0, -self.move_speed)
            if self.rect.bottom < SCREEN_HEIGHT:        
                if pressed_keys[K_k]:
                    self.rect.move_ip(0, self.move_speed)

            self.x = self.rect.centerx
            self.y = self.rect.centery
            self.dest_x = -1
            self.dest_y = -1

    def update_path_history(self, current_time):
        if (len(self.path_history) > 1000): # Only store last 5 seconds, Ticks in .05ms intervals
            self.path_history.pop(0)

        self.path_history.append([current_time, self.rect.centerx, self.rect.centery])
        
    def reset_pos(self):
        self.rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT - 100)
        self.x = self.rect.centerx
        self.y = self.rect.centery
        self.dest_x = -1
        self.dest_y = -1
 
    def draw(self, surface):
        surface.blit(self.image, self.rect)    

#################################################################################################
#################################################################################################
#################################################################################################

class Projectile(pygame.sprite.Sprite): 
    def __init__(self, x, y, player_x, player_y, theta, aim_mode):
        super().__init__() 
        self.player_x = player_x
        self.player_y = player_y
        self.initial_x = x
        self.initial_y = y
        self.theta = theta
        self.aim_mode = aim_mode
        self.x_vel = math.cos(self.theta * (2*math.pi/360)) * 7
        self.y_vel = math.sin(self.theta * (2*math.pi/360)) * 7
        self.x = self.initial_x
        self.y = self.initial_y
        self.miss_x = -1
        self.miss_y = -1
        self.game_area = pygame.Rect(0,0,SCREEN_WIDTH,SCREEN_HEIGHT)

        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (10,10))
        self.image = pygame.transform.rotate(self.image, 180)

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

    def update(self, rect, enemy, env):
        self.x += self.x_vel
        self.y += self.y_vel

        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

        if (self.theta + 90 > 360):
            self.theta = self.theta - 360

        for block in env.obstacle_group:
            if (block.rect.collidelistall([self.rect])):
                data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.rect.centerx,
                    "Projectile_y_pos": self.rect.centery, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Theta": round(self.theta, 2),
                    #"Blocked": 1,
                    "Hit": 0}
                write_csv(data)
                env.step_reward -= 10
                env.shots_taken += 1
                env.E1.reloading = 0
                self.kill()

        if (rect.collidelistall([self.rect])):

            data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.rect.centerx,
                    "Projectile_y_pos": self.rect.centery, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Theta": round(self.theta, 2),
                    #"Blocked": 0,
                    "Hit": 1}

            if (self.aim_mode == 0):
                if (enemy.neuro <= 1 - enemy.learning_rate):
                    enemy.neuro += enemy.learning_rate
            if (self.aim_mode == 1):
                if (enemy.social <= 1 - enemy.learning_rate):
                    enemy.social += enemy.learning_rate
            if (self.aim_mode == 2):
                if (enemy.cultural <= 1 - enemy.learning_rate):
                    enemy.cultural += enemy.learning_rate

            #print(enemy.neuro, enemy.social, enemy.cultural)
            write_csv(data)
            env.step_reward += self.calculate_reward(True, rect, env.P1.path_history, self.initial_x, self.initial_y)
            env.shots_taken += 1
            env.shots_hit += 1
            env.E1.reloading = 0
            self.kill()

        if (self.rect.top > rect.bottom and self.miss_x == -1 and self.miss_y == -1):
            self.miss_x = self.rect.centerx
            self.miss_y = self.rect.centery

        if not self.game_area.contains(self.rect):
            if (self.rect.top < rect.bottom):
                self.miss_x = self.rect.centerx
                self.miss_y = self.rect.centery

            data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.miss_x,
                    "Projectile_y_pos": self.miss_y, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Theta": round(self.theta, 2),
                    #"Blocked": 0,
                    "Hit": 0}

            if (self.aim_mode == 0):
                if (enemy.neuro > 0 + enemy.learning_rate):
                    enemy.neuro += -enemy.learning_rate
            if (self.aim_mode == 1):
                if (enemy.social > 0 + enemy.learning_rate):
                    enemy.social += -enemy.learning_rate
            if (self.aim_mode == 2):
                if (enemy.cultural > 0 + enemy.learning_rate):
                    enemy.cultural += -enemy.learning_rate

            #print(enemy.neuro, enemy.social, enemy.cultural)
            write_csv(data)
            env.step_reward += self.calculate_reward(False, rect, env.P1.path_history, self.initial_x, self.initial_y)
            env.shots_taken += 1
            env.E1.reloading = 0
            self.kill()
        return

    def draw(self, surface):
        surface.blit(self.image, self.rect)  

    def calculate_reward(self, hit, rect, path_history, enemy_x, enemy_y):
        reward = 0

        if hit == True:
            aimer_dist = math.sqrt((self.initial_x - rect.centerx)**2 + (self.initial_y - rect.centery)**2)

            reward = 10*(min(aimer_dist/10,5))

            reward = min(reward,50)

        else:
            proj_dist = math.sqrt((self.miss_x - rect.centerx)**2 + (self.miss_y - rect.centery)**2)
            aimer_dist = math.sqrt((self.initial_x - rect.centerx)**2 + (self.initial_y - rect.centery)**2)

            reward = -5*(proj_dist/100)*(max(100/aimer_dist,1))

            reward = max(reward,-5)

        '''if hit == True:
            reward = 50
        else:
            reward = -5'''

        return reward

#################################################################################################
#################################################################################################
#################################################################################################

class Boost(pygame.sprite.Sprite):
    def __init__(self, x, y, type, death):
        super().__init__()
        self.type = type
        if (self.type == 0):
            self.image = pygame.transform.scale(pygame.image.load("pointorb.png"), (10,10))
        if (self.type == 1):
            self.image = pygame.transform.scale(pygame.image.load("boostorb.png"), (10,10))

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.life = 0
        self.death = death

    def reset(self):
        self.kill()
        
    def update(self, player_rect, env):
        self.life += 1
        if (player_rect.collidelistall([self.rect])):
            if (self.type == 0):
                env.step_reward -= 50
            if (self.type == 1):
                if (env.P1.boosted == -1):
                    env.step_reward -= 10
                    env.P1.move_speed = env.P1.move_speed*1.5
                    env.P1.boosted = 0
            self.kill()
        if (self.life == self.death*100):
            if (self.type == 0):
                env.step_reward += 50
            if (self.type == 1):
                env.step_reward += 10
            self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)  

#################################################################################################
#################################################################################################
#################################################################################################

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, death):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load("Obstacle.png"), (random.randint(2,5)*10,10))
        self.image = pygame.transform.rotate(self.image, 180)

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.life = 0
        self.death = death

    def reset(self):
        self.kill()
 
    def update(self):
        self.life += 1

        if (self.life == self.death*100):
            self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)  