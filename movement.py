import random
import math
import sys
from projectile import *

def movement(cmd, SCREEN_WIDTH, SCREEN_HEIGHT, self, collectable_group):

    if (len(collectable_group) > 0):
        search_orb = 1
        closest_dist = 999999999
        dest_x = -1
        dest_y = -1
        for orb in collectable_group:
            orb_dist = math.sqrt((orb.rect.centerx - self.rect.centerx)**2 + (orb.rect.centery - self.rect.centery)**2)
            if (orb_dist < closest_dist):
                closest_dist = orb_dist
                target_orb = orb
                dest_x = orb.rect.centerx
                dest_y = orb.rect.centery
    else:
        search_orb = 0

    if (cmd != 'Base' and cmd != 'Neuro' and cmd != 'Social' and cmd != 'Cultural'):
        print("Please enter command in form py gameenv.py [Movement] [Engine]")
        exit()

        ############################# Random Movement ##################################
    if (cmd == 'Base'):
        if (self.dest_x == -1 and self.dest_y == -1):
            #self.dest_x = random.randint(0,SCREEN_WIDTH)
            #self.dest_y = random.randint(0,SCREEN_HEIGHT)

            self.dest_x = self.rect.centerx + random.randint(-200,200)
            self.dest_y = self.rect.centery + random.randint(-50,50)
            
            if (self.dest_x < 0):
                self.dest_x = abs(self.dest_x)
            if (self.dest_y < 0):
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

        if (self.rect.left < 0 or self.rect.right > SCREEN_WIDTH or self.rect.top < 0 or self.rect.bottom > SCREEN_HEIGHT):
            self.dest_x = -1
            self.dest_y = -1
            if (self.rect.left < 0):
                self.rect.move_ip(1,0)
            elif (self.rect.right > SCREEN_WIDTH):
                self.rect.move_ip(-1,0)
            elif (self.rect.top < 0):
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

    ############################# Teleport - Neuro ##################################
    if (cmd == 'Neuro'):
        self.dest_x = random.randint(0,SCREEN_WIDTH)
        self.dest_y = random.randint(SCREEN_HEIGHT/2,SCREEN_HEIGHT)

        line = [(self.rect.centerx, self.rect.centery),(self.dest_x, self.dest_y)]
        self.theta = getAngle(line[1], line[0])

        if (self.theta > 360):
            self.theta = self.theta - 360

        self.x_vel = math.cos(self.theta * (2*math.pi/360)) * self.move_speed
        self.y_vel = math.sin(self.theta * (2*math.pi/360)) * self.move_speed

        self.dist_travelled += abs(self.x_vel) + abs(self.y_vel)
        if (self.dist_travelled >= 200):
            self.dist_travelled = 0
            self.rect.centery = self.dest_y
            self.rect.centerx = self.dest_x

    ############################# Left and right Movement - Social ##################################
    if (cmd == 'Social'):
        if self.dest_x == -1 and self.dest_y == -1:
            self.y += self.move_speed
            if self.rect.bottom >= SCREEN_HEIGHT:
                self.dest_x = SCREEN_WIDTH
                self.dest_y = SCREEN_HEIGHT*3/4
        else:
            if (search_orb == 1):
                line = [(self.rect.centerx, self.rect.centery),(dest_x, dest_y)]
            elif (search_orb == 0):
                line = [(self.rect.centerx, self.rect.centery),(self.dest_x, self.dest_y)]
            self.theta = getAngle(line[1], line[0])

            if (self.theta > 360):
                self.theta = self.theta - 360

            self.x_vel = math.cos(self.theta * (2*math.pi/360)) * self.move_speed
            self.y_vel = math.sin(self.theta * (2*math.pi/360)) * self.move_speed

            if self.rect.left <= 0:
                self.dest_x = SCREEN_WIDTH/2
                self.dest_y = SCREEN_HEIGHT
            if self.rect.right >= SCREEN_WIDTH:
                self.dest_x = SCREEN_WIDTH/2
                self.dest_y = SCREEN_HEIGHT/2
            if self.rect.top <= SCREEN_HEIGHT/2:
                self.dest_x = 0
                self.dest_y = SCREEN_HEIGHT*3/4
            if self.rect.bottom >= SCREEN_HEIGHT:
                self.dest_x = SCREEN_WIDTH
                self.dest_y = SCREEN_HEIGHT*3/4

            self.x += self.x_vel
            self.y += self.y_vel

        self.rect.centery = int(self.y)
        self.rect.centerx = int(self.x)

    ############################# Left and right Movement Juke - Cultural ##################################
    if (cmd == 'Cultural'):
        if self.dest_x == -1 and self.dest_y == -1:
            self.y += self.move_speed
            if self.rect.bottom >= SCREEN_HEIGHT:
                self.dest_x = SCREEN_WIDTH
                self.dest_y = SCREEN_HEIGHT*3/4
        else:
            if (search_orb == 1):
                line = [(self.rect.centerx, self.rect.centery),(dest_x, dest_y)]
            elif (search_orb == 0):
                line = [(self.rect.centerx, self.rect.centery),(self.dest_x, self.dest_y)]
            self.theta = getAngle(line[1], line[0])

            if (self.theta > 360):
                self.theta = self.theta - 360

            self.x_vel = math.cos(self.theta * (2*math.pi/360)) * self.move_speed
            self.y_vel = math.sin(self.theta * (2*math.pi/360)) * self.move_speed

            if self.rect.left <= 0:
                self.dest_x = SCREEN_WIDTH/2
                self.dest_y = SCREEN_HEIGHT
                self.juke = 1
            if self.rect.right >= SCREEN_WIDTH:
                self.dest_x = SCREEN_WIDTH/2
                self.dest_y = SCREEN_HEIGHT/2
                self.juke = 1
            if self.rect.top <= SCREEN_HEIGHT/2:
                self.dest_x = 0
                self.dest_y = SCREEN_HEIGHT*3/4
                self.juke = 1
            if self.rect.bottom >= SCREEN_HEIGHT:
                self.dest_x = SCREEN_WIDTH
                self.dest_y = SCREEN_HEIGHT*3/4
                self.juke = 1

            self.x += self.x_vel*self.juke
            self.y += self.y_vel*self.juke

            if (self.juke == -1):
                self.dist_travelled += abs(self.x_vel) + abs(self.y_vel)
            if (self.dist_travelled >= 50):
                self.dist_travelled = 0
                self.juke = 1

        self.rect.centery = int(self.y)
        self.rect.centerx = int(self.x)
