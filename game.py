import pygame, sys
import math
import os.path
import csv

from re import X
from time import time
from pygame.locals import *
from projectile import *
 
pygame.init()

filename = './data.csv'
FPS = 240
FramePerSec = pygame.time.Clock()
 
# Predefined some colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
# Screen information
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 900
game_area = pygame.Rect(0,0,720,900)
 
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")

check_file = os.path.isfile(filename)
print("data.csv exists: " + str(check_file))

def write_csv(new_data):
    field_names = ["Shooter_x_pos","Shooter_y_pos",
                "Projectile_x_pos","Projectile_y_pos",
                "Player_x_pos_current","Player_y_pos_current",
                "Player_x_pos_initial","Player_y_pos_initial",
                "Theta",
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
        self.image = pygame.transform.scale(pygame.image.load("Enemy.png"), (50,50))
        self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect() 
        self.rect.center = (SCREEN_WIDTH/2, 80)
        self.x = self.rect.x 
        self.y = self.rect.y
 
    def update(self):
        pressed_keys = pygame.key.get_pressed()

        if self.rect.left > 0:
              if pressed_keys[K_a]:
                  self.rect.move_ip(-2, 0)
        if self.rect.right < SCREEN_WIDTH:        
              if pressed_keys[K_d]:
                  self.rect.move_ip(2, 0)
        if self.rect.top > 0:
              if pressed_keys[K_w]:
                  self.rect.move_ip(0, -2)
        if self.rect.bottom < SCREEN_HEIGHT:        
              if pressed_keys[K_s]:
                  self.rect.move_ip(0, 2)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 
    
#################################################################################################
#################################################################################################
#################################################################################################
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Player.png"), (50,50))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH/2, 700)
 
    def update(self):
        pressed_keys = pygame.key.get_pressed()
         
        if self.rect.left > 0:
              if pressed_keys[K_LEFT]:
                  self.rect.move_ip(-2, 0)
        if self.rect.right < SCREEN_WIDTH:        
              if pressed_keys[K_RIGHT]:
                  self.rect.move_ip(2, 0)
        if self.rect.top > 0:
              if pressed_keys[K_UP]:
                  self.rect.move_ip(0, -2)
        if self.rect.bottom < SCREEN_HEIGHT:        
              if pressed_keys[K_DOWN]:
                  self.rect.move_ip(0, 2)

    def reset_pos(self):
        self.rect.center = (SCREEN_WIDTH/2, 700)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect)    

#################################################################################################
#################################################################################################
#################################################################################################

class Projectile(pygame.sprite.Sprite): 
    def __init__(self, x, y, player_x, player_y, theta):
        super().__init__() 
        self.player_x = player_x
        self.player_y = player_y
        self.initial_x = x
        self.initial_y = y
        self.theta = theta
        self.x_vel = math.cos(self.theta * (2*math.pi/360)) * 7
        self.y_vel = math.sin(self.theta * (2*math.pi/360)) * 7
        self.x = self.initial_x
        self.y = self.initial_y
        self.miss_x = -1
        self.miss_y = -1
        self.game_area = game_area

        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (20,20))
        self.image = pygame.transform.rotate(self.image, 180)

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y

    def update(self, rect):
        self.x += self.x_vel
        self.y += self.y_vel

        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

        if (self.theta + 90 > 360):
            self.theta = self.theta - 360

        if (rect.collidelistall([self.rect])):

            data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.rect.centerx,
                    "Projectile_y_pos": self.rect.centery, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Theta": round(self.theta + 90, 2),
                    "Hit": 1}

            write_csv(data)
            self.kill()

        if (self.rect.top > rect.bottom and self.miss_x == -1 and self.miss_y == -1):
            self.miss_x = self.rect.centerx
            self.miss_y = self.rect.centery

        if not self.game_area.contains(self.rect):

            data = {"Shooter_x_pos": self.initial_x, 
                    "Shooter_y_pos": self.initial_y,
                    "Projectile_x_pos": self.miss_x,
                    "Projectile_y_pos": self.miss_y, 
                    "Player_x_pos_current": rect.centerx,
                    "Player_y_pos_current": rect.centery,
                    "Player_x_pos_initial": self.player_x,
                    "Player_y_pos_initial": self.player_y,
                    "Theta": round(self.theta + 90, 2),
                    "Hit": 0}

            write_csv(data)
            self.kill()

    def draw(self, surface):
        surface.blit(self.image, self.rect)  

#################################################################################################
#################################################################################################
#################################################################################################
         
P1 = Player()
E1 = Enemy()
projectile_group = pygame.sprite.Group()
 
while True:     
    line = [(E1.rect.centerx, E1.rect.centery),(P1.rect.centerx, P1.rect.centery)]
    theta = getAngle(line[1], line[0])
    if (theta > 360):
        theta = theta - 360
    for event in pygame.event.get():              
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bullet = Projectile(E1.rect.centerx, E1.rect.centery, P1.rect.centerx, P1.rect.centery, theta)
                projectile_group.add(bullet)
                #print(bullet.initial_x, bullet.initial_y, bullet.player_x, bullet.player_y, theta - 90)
                #print(bullet.x_vel, bullet.y_vel)
            if event.key == pygame.K_r: # Clear projectile cache
                projectile_group.empty()
            if event.key == pygame.K_p:
                P1.reset_pos()

    DISPLAYSURF.fill(WHITE)
    P1.draw(DISPLAYSURF)
    E1.draw(DISPLAYSURF)
    pygame.draw.line(DISPLAYSURF, (238, 75, 43), line[0], line[1])
    projectile_group.draw(DISPLAYSURF)
    
    P1.update()
    #E1.update()
    projectile_group.update(P1.rect)
         
    pygame.display.update()
    FramePerSec.tick(FPS)