from re import X
import pygame, sys
from pygame.locals import *
import random
import math
 
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
            bullet.update_player_loc(player.rect.x + 17, player.rect.y)

    
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
        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (20,20))
        self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 4

    def update_player_loc(self, player_x, player_y):
        self.player_x = player_x
        self.player_y = player_y

    def update(self):
        # Calculate xy distance self and player
        dist_x = self.player_x - self.initial_x
        dist_y = self.player_y - self.initial_y
        ratio = dist_x/dist_y

        if self.rect.top < SCREEN_HEIGHT:
            if (dist_x):
                if (dist_x > 0):
                    self.rect.x += 3
                elif (dist_x < 0):
                    self.rect.x += -3

            self.rect.y += 5

    
    def draw(self, surface):
        surface.blit(self.image, self.rect)  

#################################################################################################
#################################################################################################
#################################################################################################
         
P1 = Player()
E1 = Enemy()
bullet = Projectile(E1.rect.x, E1.rect.y, P1.rect.x + 17, P1.rect.centery)
 
while True:     
    for event in pygame.event.get():              
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    DISPLAYSURF.fill(WHITE)
    P1.draw(DISPLAYSURF)
    E1.draw(DISPLAYSURF)
    bullet.draw(DISPLAYSURF)
    
    P1.update()
    #E1.update(P1.rect.x, P1.rect.y)
    E1.shoot(bullet, P1)
    bullet.update()
         
    pygame.display.update()
    FramePerSec.tick(FPS)