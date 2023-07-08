import pygame, sys
import math

from re import X
from time import time
from pygame.locals import *
from projectile import *
 
pygame.init()
 
FPS = 240
FramePerSec = pygame.time.Clock()
 
# Predefined some colors
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
# Screen information
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
 
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")

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
 
    def update(self, player_x, player_y):
        self.image = pygame.transform.rotate(self.image, 180)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 

    def shoot(self, bullet, player):
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_SPACE]:
            bullet.rect.center = (self.rect.center)
            bullet.update_player_loc(player.rect.centerx, player.rect.centery)

    
#################################################################################################
#################################################################################################
#################################################################################################
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Player.png"), (50,50))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH/2, 520)
 
    def update(self):
        pressed_keys = pygame.key.get_pressed()
       #if pressed_keys[K_UP]:
            #self.rect.move_ip(0, -5)
       #if pressed_keys[K_DOWN]:
            #self.rect.move_ip(0,5)
         
        if self.rect.left > 0:
              if pressed_keys[K_LEFT]:
                  self.rect.move_ip(-2, 0)
        if self.rect.right < SCREEN_WIDTH:        
              if pressed_keys[K_RIGHT]:
                  self.rect.move_ip(2, 0)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect)    

#################################################################################################
#################################################################################################
#################################################################################################

class Projectile(pygame.sprite.Sprite): 
    def __init__(self, x, y, player_x, player_y):
        super().__init__() 
        self.player_x = player_x
        self.player_y = player_y
        self.initial_x = x
        self.initial_y = y
        self.angle = 0

        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (20,20))
        self.image = pygame.transform.rotate(self.image, 180)

        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.speed = 17
        self.time = 0.0

    def update_player_loc(self, player_x, player_y):
        self.player_x = player_x
        self.player_y = player_y
        self.angle = self.findAngle(self.player_x, self.player_y, self.initial_x, self.initial_y)
        self.time = 0

    def update(self):
        new_pos = self.calculatePath(self.rect.centerx, self.rect.centery, self.angle)
        self.rect.x = new_pos[0]
        self.rect.y = new_pos[1]

    def findAngle(self, enemy_x, enemy_y, player_x, player_y): # HELP FIGURING OUT ANGLE!!! AND PATH CALCULATION!!!
        try:
            angle = math.atan((enemy_y - player_y) / (enemy_x - player_x))
        except:
            angle = math.pi / 2

        if player_y < enemy_y and player_x > enemy_x:
            angle = abs(angle)
        elif player_y < enemy_y and player_x < enemy_x:
            angle = math.pi - angle
        elif player_y > enemy_y and player_x < enemy_x:
            angle = math.pi + abs(angle)
        elif player_y > enemy_y and player_x > enemy_x:
            angle = (math.pi * 2) - angle
        print(angle)
        return angle

    def calculatePath(self, initial_x, initial_y, angle):
        vel_x = math.cos(angle)
        vel_y = math.sin(angle)

        move_x = vel_x
        move_y = vel_y

        new_x = round(move_x + initial_x)
        new_y = round(initial_y - move_y)

        return(new_x, new_y)

    
    def draw(self, surface):
        surface.blit(self.image, self.rect)  

#################################################################################################
#################################################################################################
#################################################################################################
         
P1 = Player()
E1 = Enemy()
bullet = Projectile(E1.rect.centerx, E1.rect.centery, P1.rect.centerx, P1.rect.centery)
 
while True:     
    line = [(E1.rect.centerx, E1.rect.centery),(P1.rect.centerx, P1.rect.centery)]
    for event in pygame.event.get():              
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    DISPLAYSURF.fill(WHITE)
    P1.draw(DISPLAYSURF)
    E1.draw(DISPLAYSURF)
    pygame.draw.line(DISPLAYSURF, (238, 75, 43), line[0], line[1])
    bullet.draw(DISPLAYSURF)
    
    P1.update()
    #E1.update(P1.rect.x, P1.rect.y)
    E1.shoot(bullet, P1)
    bullet.update()
         
    pygame.display.update()
    FramePerSec.tick(FPS)