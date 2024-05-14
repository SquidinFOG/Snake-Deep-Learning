import torch
import random
import numpy as np
from collections import deque
from game import *
from model import *
import os
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
# pour resoudre le pb "Initializing libiomp5md.dll, 
# but found libiomp5md.dll already initialized."



MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001                  #Learning rate



class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        #self.model = Linear_Net(11, 256, 3)                         # 11 input (état), 256 hidden (boite noire), 3 output 
        #self.model = Linear_Net(11, 128, 3)                         
        #self.model = Linear_Net(11, 64, 3)
        self.model = Linear_Net2(11, 128,64, 3)
        #self.model = Linear_Net3(11,256, 256,128, 3)
        #self.model = Linear_Net3(11,32, 64,32, 3)
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma) #Modèle


    def get_state(self, game):
        #recupere l'état de la partie (<=> get_input dans les génétiques)
        head = game.snake[0]
        point_g = Point(head.x - 20, head.y)
        point_d = Point(head.x + 20, head.y)
        point_h = Point(head.x, head.y - 20)
        point_b = Point(head.x, head.y + 20)
        
        dir_g = game.direction == Direction.LEFT
        dir_d = game.direction == Direction.RIGHT
        dir_h = game.direction == Direction.UP
        dir_b = game.direction == Direction.DOWN

        state = [
            # Danger devant
            (dir_d and game._collision(point_d)) or 
            (dir_g and game._collision(point_g)) or 
            (dir_h and game._collision(point_h)) or 
            (dir_b and game._collision(point_b)),

            # Danger droite
            (dir_h and game._collision(point_d)) or 
            (dir_b and game._collision(point_g)) or 
            (dir_g and game._collision(point_h)) or 
            (dir_d and game._collision(point_b)),

            # Danger gauche
            (dir_b and game._collision(point_d)) or 
            (dir_h and game._collision(point_g)) or 
            (dir_d and game._collision(point_h)) or 
            (dir_g and game._collision(point_b)),
            
            # la direction "hot-one encoded" : [1,0,0,0],[0,1,0,0],[0,0,1,0] ou [0,0,0,1]
            dir_g,
            dir_d,
            dir_h,
            dir_b,
            
            # Position de la pomme 
            game.food.x < game.head.x,  # 1 si pomme à gauche               0sinon
            game.food.x > game.head.x,  #  1 si pomme à droite              """"""
            game.food.y < game.head.y,  # 1 si pomme au dessus              """"""
            game.food.y > game.head.y  # 1 si pomme en dessous              """"""
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # on supprime le plus vieux "souvenir" si MAX_MEMORY est atteint

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # liste de tuples
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample) 
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        #entrainement sur juste un seul état
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # au debut il faut des moves randoms car l'agent "tatonne" on utilise epsi et 
        # sa valeur diminue au fur et à mesure que l'agent sentraine
        # on le fait diminuer        
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:                   #move randon
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)         #move determiner par le rézo"
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    scores = []
    scores_moy = []
    score_glisse = []
    tot_glisse = 0
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (rejoue contre les parties précedentes), plot des resultat
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            scores.append(score)
            total_score += score
            scores_moy.append(total_score / agent.n_games)
            tot_glisse += score
            if agent.n_games > 10 : 
                tot_glisse -= scores[-10]
            score_glisse.append(tot_glisse/10)
            #plot(scores, scores_moy)
            plot2(score_glisse)

def plot(scores, scores_moy):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.xlabel('Nombre de parties')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(scores_moy)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(scores_moy)-1, scores_moy[-1], str(scores_moy[-1]))
    plt.show(block=False)
    plt.pause(.1)

def plot2(scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.xlabel('Nombre de parties')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.show(block=False)
    plt.pause(.1)



if __name__ == '__main__':
    train()