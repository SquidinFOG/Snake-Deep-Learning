from Game import *
import neat

taille_partie=150 # Taille d'une partie (pixels)
taille_grille=20 # Taille de la grille d'une partie
fps=100

# Attention !! s'assurer que cela correspond à la taille de la population
nrow=5 # Nombre de colonnes et lignes pour l'aggichage
ncol=10


# Initialisation de la fenêtre pour l'affichage
pygame.init()
pygame.event.pump()
screen = pygame.display.set_mode((ncol*taille_partie, nrow*taille_partie))
clock = pygame.time.Clock()


# Fonction d'évaluation des populations
def eval_genomes(genomes, config):
    parties={}
    nets={}
    for genome_id, genome in genomes:
        parties[genome_id] = Game(taille_grille)
        nets[genome_id] = neat.nn.FeedForwardNetwork.create(genome, config)
            
    continuer = True
    while (len(parties)!=0) and continuer:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                continuer=False
                pygame.quit()

        screen.fill((0,0,0))
        del_item=[]
        i=-1
        for genome_id, genome in genomes:
            i+=1
            if genome_id not in parties.keys():# Si la partie est déjà finie, on passe 
                continue
            # on affiche la partie
            parties[genome_id].draw(screen, i%ncol, i//ncol, taille_partie)
            
            # on applique le réseau aux inputs
            state=parties[genome_id].input_return1()
            output = nets[genome_id].activate(state)
            output = output.index(max(output))

            # on fait avvancer le serpent
            alive = parties[genome_id].step(output)
            if not alive[0]:
                del_item.append(genome_id)
                genome.fitness = 35*alive[1]+alive[3]
        
        # On supprime les parties finies
        for i in del_item:
            parties.pop(i)

        pygame.display.flip()
        clock.tick(fps)


def run_checkpoint(checkpoint, nb_gen=1, checkpointer=None):
    p = neat.Checkpointer.restore_checkpoint(checkpoint)
    if checkpointer:
        p.add_reporter(neat.Checkpointer(checkpointer))
    return p.run(eval_genomes, nb_gen)


if __name__=='__main__':
    # Création de l'objet config
    config = neat.Config(neat.DefaultGenome,
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet,
                        neat.DefaultStagnation,
                        "config.txt")

    # Création du neat et run depuis le début
    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10)) #Checkpoint toute les 10 générations
    p.run(eval_genomes,1000)