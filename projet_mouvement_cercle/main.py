#! /usr/bin/env python
# Time-stamp: <2019-03-09 09:45:17 christophe@pallier.org>

""" Draw a circle using pygame (see <http://www.pygame.org>). """

import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (127, 127, 127)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()
screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
screen.fill(WHITE)
W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
clock = pygame.time.Clock()
                
# display the backbuffer
pygame.display.flip()

# wait till the window is closed by escape or q pressed
while 1:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
                break
        elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.unicode == 'q':
                        break
        
        screen.fill(WHITE)
        
        mouse_position = pygame.mouse.get_pos()
        
        pygame.draw.circle(screen, RED, (mouse_position[0],mouse_position[1]), 30, 0)
        pygame.display.flip()
        
        clock.tick(60)
pygame.quit() 
