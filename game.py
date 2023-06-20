import pygame, sys
from pygame.locals import *
import random
 
pygame.init()
 
FPS = 240
FramePerSec = pygame.time.Clock()
 
# Predefined some colors
#BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
 
# Screen information
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
 
DISPLAYSURF = pygame.display.set_mode((400,600))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Game")
 
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Enemy.png"), (50,50))
        self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect() 
        self.rect.center = (200, 80)
 
    def update(self):
        self.image = pygame.transform.rotate(self.image, 180)
 
    def draw(self, surface):
        surface.blit(self.image, self.rect) 

    def shoot(self, bullet):
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_SPACE]:
            bullet.rect.center = (200, 80)

    
 
 
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.image = pygame.transform.scale(pygame.image.load("Player.png"), (50,50))
        self.rect = self.image.get_rect()
        self.rect.center = (200, 520)
 
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

class Projectile(pygame.sprite.Sprite): 
    def __init__(self):
        self.image = pygame.transform.scale(pygame.image.load("Bullet.png"), (20,20))
        self.image = pygame.transform.rotate(self.image, 180)
        self.rect = self.image.get_rect()
        self.rect.center = (200, 80)

    def update(self):
        if self.rect.bottom > 50:
            self.rect.move_ip(0,5)
    
    def draw(self, surface):
        
        surface.blit(self.image, self.rect)  
         
P1 = Player()
E1 = Enemy()
bullet = Projectile()
 
while True:     
    for event in pygame.event.get():              
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    P1.update()
    #E1.update()
    E1.shoot(bullet)
    bullet.update()
     
    DISPLAYSURF.fill(WHITE)
    P1.draw(DISPLAYSURF)
    E1.draw(DISPLAYSURF)
    bullet.draw(DISPLAYSURF)

         
    pygame.display.update()
    FramePerSec.tick(FPS)