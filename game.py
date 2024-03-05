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
from tcav import run_tcav
 
pygame.init()
clock = pygame.time.Clock()
current_time = 0
collectable_time = 0
obstacle_time = 0
collectable_spawn_time = random.randint(2,5)
obstacle_spawn_time = random.randint(7,10)

filename = './human_hist.csv'
filename2 = './tcav.csv'
FPS = 240
FramePerSec = pygame.time.Clock()

SPRITE_SIZE = 25
BULLET_SIZE = 10
 
# Predefined some colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
# Screen information
SCREEN_WIDTH = 360
SCREEN_HEIGHT = 450
game_area = pygame.Rect(0,0,360,450)
 
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")

check_file = os.path.isfile(filename)
print("human.csv exists: " + str(check_file))
check_file2 = os.path.isfile(filename2)
print("tcav.csv exists: " + str(check_file2))

def write_csv(new_data):
    print("Hello")
    field_names = ["Shooter_x_pos","Shooter_y_pos",
                "Projectile_x_pos","Projectile_y_pos",
                "Player_x_pos_current","Player_y_pos_current",
                "Player_x_pos_initial","Player_y_pos_initial",
                "Distance_x","Distance_y",
                "Displacement_x","Displacement_y",
                "Theta",
                #"Blocked",
                "Hit"]
    dict = new_data
    with open(filename, 'a') as file:
        dict_object = csv.DictWriter(file, fieldnames=field_names, lineterminator = '\n') 
        dict_object.writerow(dict)

def write_csv2(new_data):
    field_names = ["Shooter_x_pos","Shooter_y_pos",
                "Projectile_x_pos","Projectile_y_pos",
                "Player_x_pos_current","Player_y_pos_current",
                "Player_x_pos_initial","Player_y_pos_initial",
                "Distance_x","Distance_y",
                "Displacement_x","Displacement_y",
                "Theta",
                "Hit"]
    dict = new_data
    with open(filename2, 'a') as file:
        dict_object = csv.DictWriter(file, fieldnames=field_names, lineterminator = '\n') 
        dict_object.writerow(dict)

def calculate_distance(path_history):

    if len(path_history) == 0:
        return 0,0,0,0

    if len(path_history) < 400:
        displacement_x = path_history[len(path_history)-1][1]-path_history[0][1]
        displacement_y = path_history[len(path_history)-1][2]-path_history[0][2]

        distance_x = 0
        distance_y = 0
        for i in range(0,len(path_history)-1):
            distance_x += abs(path_history[i+1][1]-path_history[i][1])  
            distance_y += abs(path_history[i+1][2]-path_history[i][2])
    else:
        displacement_x = path_history[len(path_history)-1][1]-path_history[len(path_history)-400][1]
        displacement_y = path_history[len(path_history)-1][2]-path_history[len(path_history)-400][2]

        distance_x = 0
        distance_y = 0
        for i in range(len(path_history)-400,len(path_history)-1):
            distance_x += abs(path_history[i+1][1]-path_history[i][1])  
            distance_y += abs(path_history[i+1][2]-path_history[i][2])

    return distance_x, distance_y, displacement_x, displacement_y

#################################################################################################
#################################################################################################
#################################################################################################
 
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Enemy.png"), (SPRITE_SIZE,SPRITE_SIZE))
        #self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect() 
        self.rect.center = (SCREEN_WIDTH/2, 80)
        self.x = self.rect.centerx 
        self.y = self.rect.centery
        self.theta = 90
        self.projectile_range = 800
        self.sensitivity = 1
        self.move_speed = 2
        self.aim_mode = 3 # 0 - Neuro, 1 - Social, 2 - Cultural, 3 - Player
        self.aim_text = 'Player'
        self.auto = 0

        self.neuro = 0.75
        self.social = 0.75
        self.cultural = 0.5
        self.learning_rate = 0.05
 
    def update(self, theta):
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
        '''neuro_x = path_history[0][1] - player_x
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

        print(neuro_weight, social_weight, cultural_weight)

        if (neuro_weight >= max(social_weight, cultural_weight)):
            self.aim_mode = 0
            self.aim_text = 'Neuro'
        elif (social_weight >= cultural_weight):
            self.aim_mode = 1
            self.aim_text = 'Social'
        else:
            self.aim_mode = 2
            self.aim_text = 'Cultural'''

        self.aim_mode = random.randint(0,3)


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

    def tcav_set(self,x,y,theta):
        self.rect.centerx = x
        self.rect.centery = y
        self.x = self.rect.centerx 
        self.y = self.rect.centery
        self.theta = theta
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 
    
#################################################################################################
#################################################################################################
#################################################################################################
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Player.png"), (SPRITE_SIZE,SPRITE_SIZE))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT - 100)
        self.dest_x = -1
        self.dest_y = -1
        self.x = self.rect.centerx
        self.y = self.rect.centery
        self.x_vel = 0
        self.y_vel = 0
        self.theta = 0
        self.move_speed = 2
        self.path_history = []
        self.mode = 0
        self.move_right = 1
        self.boosted = 0
 
    def update(self, current_time):
        self.update_path_history(current_time)
        
        if (self.mode == 0):

############################# Random Movement ##################################

            if (self.dest_x == -1 and self.dest_y == -1):
                #self.dest_x = random.randint(0,SCREEN_WIDTH)
                #self.dest_y = random.randint(0,SCREEN_HEIGHT)

                self.dest_x = self.rect.centerx + random.randint(-200,200)
                self.dest_y = self.rect.centery + random.randint(-50,50)
                
                if (self.dest_x < 0):
                    self.dest_x = abs(self.dest_x)
                if (self.dest_y < SCREEN_HEIGHT/2):
                    self.dest_y = abs(self.dest_y)
                if (self.dest_x > SCREEN_WIDTH):
                    self.dest_x += -random.randint(0,150)
                if (self.dest_y > SCREEN_HEIGHT):
                    self.dest_y += -random.randint(0,50)

                line = [(self.rect.centerx, self.rect.centery),(self.dest_x, self.dest_y)]
                self.theta = getAngle(line[1], line[0])

                if (self.theta > 360):
                    self.theta = self.theta - 360

                self.x_vel = math.cos(self.theta * (2*math.pi/360)) * self.move_speed
                self.y_vel = math.sin(self.theta * (2*math.pi/360)) * self.move_speed

            if (self.rect.left < 0 or self.rect.right > SCREEN_WIDTH or self.rect.top < SCREEN_HEIGHT/2 or self.rect.bottom > SCREEN_HEIGHT):
                self.dest_x = -1
                self.dest_y = -1
                if (self.rect.left < 0):
                    self.rect.move_ip(1,0)
                elif (self.rect.right > SCREEN_WIDTH):
                    self.rect.move_ip(-1,0)
                elif (self.rect.top < SCREEN_HEIGHT/2):
                    self.rect.move_ip(0,1)
                elif (self.rect.bottom > SCREEN_HEIGHT):
                    self.rect.move_ip(0,-1)
            elif (self.rect.centerx != self.dest_x and self.rect.centery != self.dest_y):
                self.x += self.x_vel
                self.y += self.y_vel

                self.rect.centerx = int(self.x)
                self.rect.centery = int(self.y)
            else:
                self.dest_x = -1
                self.dest_y = -1

############################# Left and right Movement ##################################

            '''if self.move_right == 1:
                self.x += self.move_speed
            if self.move_right == 0:
                self.x -= self.move_speed
            if self.rect.left <= 0:
                self.move_right = 1
            if self.rect.right >= SCREEN_WIDTH:
                self.move_right = 0
            self.rect.centerx = int(self.x)'''

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

    def tcav_set(self,x,y):
        self.rect.centerx = x
        self.rect.centery = y
        self.x = self.rect.centerx
        self.y = self.rect.centery

    def tcav_update(self,end_x,end_y,start_x,start_y, length): # Todo, implement movement 2s prior
        engine_mode = False

        line = [(start_x, start_y),(end_x, end_y)]
        theta = getAngle(line[1], line[0])

        if (theta > 360):
            theta = theta - 360

        x_vel = math.cos(theta * (2*math.pi/360)) * self.move_speed
        y_vel = math.sin(theta * (2*math.pi/360)) * self.move_speed
        
        self.x += x_vel
        self.y += y_vel

        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

        if (length == 0):
            engine_mode = True

        return engine_mode
 
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
        self.game_area = game_area

        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (BULLET_SIZE,BULLET_SIZE))
        self.image = pygame.transform.rotate(self.image, 180)

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

    def update(self, rect, enemy, path_history):
        self.x += self.x_vel
        self.y += self.y_vel

        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

        if (self.theta + 90 > 360):
            self.theta = self.theta - 360

        distance_x, distance_y, displacement_x, displacement_y = calculate_distance(path_history)

        for block in obstacle_group:
            if (block.rect.collidelistall([self.rect])):
                data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.rect.centerx,
                    "Projectile_y_pos": self.rect.centery, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Distance_x": distance_x,
                    "Distance_y": distance_y,
                    "Displacement_x": displacement_x,
                    "Displacement_y": displacement_y,
                    "Theta": round(self.theta, 2),
                    #"Blocked": 1,
                    "Hit": 0}
                write_csv(data)
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
                    "Distance_x": distance_x,
                    "Distance_y": distance_y,
                    "Displacement_x": displacement_x,
                    "Displacement_y": displacement_y,
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
                    "Distance_x": distance_x,
                    "Distance_y": distance_y,
                    "Displacement_x": displacement_x,
                    "Displacement_y": displacement_y,
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

            print(enemy.neuro, enemy.social, enemy.cultural)
            write_csv(data)
            self.kill()
        return

    def draw(self, surface):
        surface.blit(self.image, self.rect)  

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
        
    def update(self, player_rect, player):
        self.life += 1
        if (player_rect.collidelistall([self.rect])):
            if (self.type == 0):
                reward = 0
            if (self.type == 1):
                if (player.boosted == 0):
                    player.move_speed = player.move_speed*1.5
                    player.boosted = 1
            self.kill()
        if (self.life == self.death*100):
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

#################################################################################################
#################################################################################################
#################################################################################################
         
P1 = Player()
E1 = Enemy()
projectile_group = pygame.sprite.Group()
collectable_group = pygame.sprite.Group()
obstacle_group = pygame.sprite.Group()

font = pygame.font.SysFont(None,16)
text = font.render('Aim mode: ' + str(E1.aim_mode), True, BLACK)
textRect = text.get_rect()
textRect.center = (50, 50)
auto_shoot = 0
spawn_collectable = 1
spawn_obstacle = 1

engine_mode = True
tcav_initialized = False
tcav = []
coef = []
 
while True:     
    if engine_mode == True:
        text = font.render('Aim mode: ' + str(E1.aim_text), True, BLACK)
        aim_x, aim_y = E1.aim_calc(P1.rect.centerx, P1.rect.centery, P1.path_history)
        line = [(E1.rect.centerx, E1.rect.centery),(aim_x, aim_y)]
        theta = getAngle(line[1], line[0])
        if auto_shoot == 1:
            bullet = E1.take_shot(P1.rect.centerx, P1.rect.centery, P1.path_history, theta)
            projectile_group.add(bullet)
        if (theta > 360):
            theta = theta - 360
        for event in pygame.event.get():              
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    if auto_shoot == 0:
                        auto_shoot = 1
                        print(auto_shoot)
                    elif auto_shoot == 1:
                        auto_shoot = 0
                if event.key == pygame.K_SPACE:
                    bullet = E1.take_shot(P1.rect.centerx, P1.rect.centery, P1.path_history, theta)
                    projectile_group.add(bullet)
                if event.key == pygame.K_r: # Clear projectile cache
                    projectile_group.empty()
                if event.key == pygame.K_p: # Reset position
                    P1.reset_pos()
                if event.key == pygame.K_o: # Target Control
                    if (P1.mode == 0):
                        P1.mode = 1
                    else:
                        P1.mode = 0
                if event.key == pygame.K_m: # Aimer control
                    if (E1.auto == 0):
                        E1.auto = 1
                    else:
                        E1.auto = 0
                if event.key == pygame.K_t: # Train TCAV
                    tcav, coef = run_tcav()
                    engine_mode = False
                    tcav_initialized = False
                if event.key == pygame.K_x: # Reset
                    engine_mode = False
                    tcav_initialized = False
                if event.key == pygame.K_z: # Save TCAV
                    data = {"Shooter_x_pos": coef[0], 
                    "Shooter_y_pos": coef[1],
                    "Projectile_x_pos": coef[2],
                    "Projectile_y_pos":coef[3], 
                    "Player_x_pos_current": coef[4],
                    "Player_y_pos_current": coef[5],
                    "Player_x_pos_initial": coef[6],
                    "Player_y_pos_initial": coef[7],
                    "Distance_x": coef[8],
                    "Distance_y": coef[9],
                    "Displacement_x": coef[10],
                    "Displacement_y": coef[11],
                    "Theta": coef[12],
                    "Hit": 0}

                    write_csv2(data)

        current_time = pygame.time.get_ticks()

        if (collectable_time == collectable_spawn_time*100):
            spawn_collectable = 0
        if (spawn_collectable == 1):
            collectable_time += 1
        if (spawn_collectable == 0):
            orb = Boost(random.randint(0,SCREEN_WIDTH),random.randint(SCREEN_HEIGHT/2,SCREEN_HEIGHT),random.randint(0,1), random.randint(5,7))
            collectable_group.add(orb)
            spawn_collectable = 1
            collectable_spawn_time = random.randint(2,5)
            collectable_time = 0

        if (obstacle_time == obstacle_spawn_time*100):
            spawn_obstacle = 0
        if (spawn_obstacle == 1):
            obstacle_time += 1
        if (spawn_obstacle == 0):
            block = Obstacle(random.randint(0,SCREEN_WIDTH),random.randint(int(SCREEN_HEIGHT/4),SCREEN_HEIGHT/2), random.randint(2,5))
            obstacle_group.add(block)
            spawn_obstacle = 1
            obstacle_spawn_time = random.randint(7,10)
            obstacle_time = 0

        P1.update(current_time)
        E1.update(theta)
        projectile_group.update(P1.rect, E1, P1.path_history)
        collectable_group.update(P1.rect, P1)
        obstacle_group.update()
    if engine_mode == False: # Todo implement movement 2s prior
        aim_x, aim_y = E1.aim_calc(P1.rect.centerx, P1.rect.centery, P1.path_history)
        line = [(E1.rect.centerx, E1.rect.centery),(aim_x, aim_y)]
        pygame.draw.line(DISPLAYSURF, (238, 75, 43), line[0], line[1])
        if tcav_initialized == False:
            E1.tcav_set(tcav[0],tcav[1],tcav[12])
            bullet = E1.take_shot(tcav[6],tcav[7],[],tcav[12])
            projectile_group.add(bullet)
            P1.tcav_set(tcav[6],tcav[7])
            tcav_initialized = True
        else:
            engine_mode = P1.tcav_update(tcav[4],tcav[5],tcav[6],tcav[7],len(projectile_group))
            projectile_group.update(P1.rect, E1, P1.path_history)

    DISPLAYSURF.fill(WHITE)
    DISPLAYSURF.blit(text, textRect)
    P1.draw(DISPLAYSURF)
    E1.draw(DISPLAYSURF)
    pygame.draw.line(DISPLAYSURF, (238, 75, 43), line[0], line[1])
    projectile_group.draw(DISPLAYSURF)
    collectable_group.draw(DISPLAYSURF)
    obstacle_group.draw(DISPLAYSURF)
    
    if (E1.auto == 1):
        if (len(P1.path_history) > 1000):
            E1.strategize(P1.rect.centerx, P1.rect.centery, P1.path_history)
         
    pygame.display.update()
    FramePerSec.tick(FPS)