import pygame
import numpy as np
import random


class Game:
    mat_rot_right = np.array([[0,1], [-1,0]])
    mat_rot_left = np.array([[0,-1], [1,0]])

    def __init__(self, taille_grille):
        self.taille_grille = taille_grille
        self.snake = np.array([[int(taille_grille/4)+2, int(taille_grille/4)+1, int(taille_grille/4)],
                              [int(taille_grille/2), int(taille_grille/2), int(taille_grille/2)]])
        self.cases_parcourues = self.snake
        self.generate_Apple()
        self.direction = np.array([[1],[0]])
        self.iteration = 0
        self.bonus = 0
        self.mat_transform = np.array([[1,0],[0,1]]) # Matrice pour passer du repère de la partie à celui du serpent
        self.tete = pygame.image.load('./tete_snake.jpg').convert()


    def move(self):
        self.iteration+=1
        eat = True
        if not (self.snake[:,0]+np.reshape(self.direction, (2))==self.apple).all(): # Si la tête n'arrive pas sur la pomme, on retire le bout de la queue
            self.snake = self.snake[:,:-1]
            eat=False
        
        prev_distance = np.linalg.norm(self.apple-self.snake[:,0])
        self.snake = np.concatenate(([self.snake[:,0]+np.reshape(self.direction, (2))], self.snake.T), axis=0).T
        curr_distance = np.linalg.norm(self.apple-self.snake[:,0])

        if prev_distance >= curr_distance:
            self.bonus+=1
        else:
            self.bonus-= 2

        if self.snake[:,0] in self.cases_parcourues[:,:-1]:
            self.bonus -= 500
        
        # On vérifie si le serpent parcours une nouvelle case
        if self.snake[:,0].T.tolist() not in self.cases_parcourues.T.tolist():
            self.cases_parcourues = np.concatenate(([self.snake[:,0]], self.cases_parcourues.T), axis=0).T
        return eat

    def generate_Apple(self):
        self.apple = np.array([random.randint(0,self.taille_grille-1), random.randint(0,self.taille_grille-1)])
        while self.apple.tolist() in self.snake.T.tolist():
            self.apple = np.array([random.randint(0,self.taille_grille-1), random.randint(0,self.taille_grille-1)])
        

    def alive(self):
        alive = True
        if not ((0<=self.snake[:,0]).all() and (self.snake[:,0]<self.taille_grille).all()): # Si la tête est hors de la grille
            alive = False
        
        if self.snake[:,0].tolist() in self.snake[:,1:].T.tolist(): # Si le serpent se mange la queue
            alive = False

        if self.iteration ==  200: #0.5*self.taille_grille**2: # Si le serpent tourne en rond trop longtemps
            alive= False

        return alive
    
    def change_direction(self, new_dir):
        # 0 : gauche
        # 1 : tout droit
        # 2 : droite

        if new_dir==0:
            self.direction = np.dot(Game.mat_rot_left, self.direction)
            self.mat_transform = np.dot(Game.mat_rot_left, self.mat_transform)    
        elif new_dir == 2:
            self.direction = np.dot(Game.mat_rot_right, self.direction)
            self.mat_transform = np.dot(Game.mat_rot_right, self.mat_transform)    
        
    
    def draw(self, window, x, y, taille):
        cote_carre = int(taille/self.taille_grille) - 2
        taille_cellule = int(taille/self.taille_grille)

        x=x*taille
        y=y*taille

        # On dessinde le cadre
        pygame.draw.rect(window, (255,255,255), pygame.Rect(x,y,taille,taille), 2)

        # On dessine la pomme
        pygame.draw.rect(window, (255,0,0), pygame.Rect(int(x+self.apple[0]*taille_cellule+1), int(y+self.apple[1]*taille_cellule+1), cote_carre, cote_carre))

        # On dessine le serpent
        i=0
        for pos in self.snake.T:
            if i==0:
                i+=1
                window.blit(pygame.transform.smoothscale(self.tete, (cote_carre,cote_carre)), (x+pos[0]*taille_cellule+1, y+pos[1]*taille_cellule+1))
            else:
                pygame.draw.rect(window, (0,255,0), pygame.Rect(x+pos[0]*taille_cellule+1, y+pos[1]*taille_cellule+1, cote_carre, cote_carre))


    def step(self, output):
        result = [True]

        self.change_direction(output)

        if self.move():
            self.iteration=0
            self.generate_Apple()
            self.cases_parcourues=np.reshape(self.snake[:,0],(2,1))
 
        if not self.alive():
            result =  [False, self.snake.shape[1], self.cases_parcourues.shape[1], self.bonus]
        
        return result
    
    def input_return1(self):

        directions = [# Les septs directions dans le repère du serpent (sans regarder vers l'arrière)
                      np.reshape(np.dot(self.mat_transform,np.array([[1],[0]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[1],[1]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[0],[1]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[-1],[1]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[-1],[-1]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[0],[-1]])), (2)),
                      np.reshape(np.dot(self.mat_transform,np.array([[1],[-1]])), (2))
                      ]
        
        apple_direction = np.dot(self.mat_transform,
                                 np.reshape(self.apple-self.snake[:,0], (2,1))) # Direction de la pomme dans le repère du serpent
        
        # Distance de la tete à un obstacle dans chaque direction
        distances = []
        for d in directions:
            curr_pos = np.copy(self.snake[:,0])
            i=0
            while ((0<=curr_pos).all() and (curr_pos<self.taille_grille).all()):
                if curr_pos.tolist() in self.snake[:,1:].T.tolist():
                    break
                i+=1
                curr_pos+=d
            distances.append(i)
            
        return distances+np.reshape(apple_direction, (2)).tolist()
