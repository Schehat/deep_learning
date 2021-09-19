import os

import neat
import pygame

from base import Base
from bird import Bird
from config import (
    BASE_IMG,
    BG_IMG,
    BLACK,
    CLOCK,
    DRAW_LINES,
    FPS,
    WINDOW,
    WINDOW_SIZE,
    font_score,
    gen,
)
from pipe import Pipe


def draw_window(birds, pipes, base, score, gen, pipe_ind):
    WINDOW.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw()
    base.draw()
    text_score = font_score.render(f"Score: {score}", 1, BLACK)
    WINDOW.blit(text_score, (10, 10))

    text_gen = font_score.render(f"Gen: {gen}", 1, BLACK)
    WINDOW.blit(text_gen, (10, 50))

    text_alive = font_score.render(f"Alive: {len(birds)}", 1, BLACK)
    WINDOW.blit(text_alive, (10, 90))

    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(
                    WINDOW,
                    (255, 0, 0),
                    (
                        int(bird.x + bird.img.get_width() / 2),
                        int(bird.y + bird.img.get_height() / 2),
                    ),
                    (
                        int(
                            pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2
                        ),
                        pipes[pipe_ind].height,
                    ),
                    1,
                )
                pygame.draw.line(
                    WINDOW,
                    (255, 0, 0),
                    (
                        int(bird.x + bird.img.get_width() / 2),
                        int(bird.y + bird.img.get_height() / 2),
                    ),
                    (
                        int(
                            pipes[pipe_ind].x
                            + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2
                        ),
                        pipes[pipe_ind].bottom,
                    ),
                    1,
                )
            except TypeError and IndexError:
                pass

        bird.draw()

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    running the simulation of the current generation and evaluating
    their fitness level by how war far the distance is they travel
    """

    # counting generations
    global gen
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    ge = []
    nets = []
    birds = []

    #  setting up neural networks
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ge.append(genome)
        nets.append(net)
        birds.append(Bird(170, 100))

    pipes = [Pipe(WINDOW_SIZE[0])]
    base = Base(WINDOW_SIZE[0] - BASE_IMG.get_height())
    score = 0

    run = True

    while run:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # necessary with multiple pipes which pipe to put as an input
        pipe_ind = 0
        if len(birds) > 0:
            if (
                len(pipes) > 1
                and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width()
            ):
                pipe_ind = 1
        else:
            run = False

        # give each bird every frame a little fitness for staying alive
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.2

            # giving input to determine neural network to decide weather to jump or not
            output = nets[x].activate(
                (
                    bird.y,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom),
                )
            )

            #  usage of tanh activation function so result will be between -1 and 1. if over 0.5 jump
            if output[0] > 0.5:  # 0 due to only one output
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            pipe.move()
            # check collision
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 5  # punish for colliding
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            # pipes out of the screen get added to this list and removed
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        for r in rem:
            pipes.remove(r)

        if add_pipe:
            # genome gets rewarded for passing through a pipe
            for g in ge:
                g.fitness += 5
            score += 1
            pipes.append(Pipe(WINDOW_SIZE[0]))

        for x, bird in enumerate(birds):
            if (
                abs(bird.y - pipes[0].height) < 200
                or abs(bird.y - pipes[0].bottom) < 150
            ):
                ge[x].fitness += 0.1
            else:
                ge[x].fitness -= 1
            if (
                bird.y < pipes[0].bottom
                and bird.y + bird.img.get_height() > pipes[0].top
            ):
                pass
            else:
                ge[x].fitness += 0.01

        # pop birds touching the bottom & top
        for x, bird in enumerate(birds):
            if (
                bird.y + bird.img.get_height() > WINDOW_SIZE[1] - BASE_IMG.get_height()
                or bird.y < 0
            ):
                ge[x].fitness -= 5
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(birds, pipes, base, score, gen, pipe_ind)


def run(config_path):
    """
    runs the NEAT algorithm to train the neural network
    :param config_path: location of config file
    :return: None
    """
    # load the config
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # setting population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 1. simulation & 2. amount of generations here limitless
    # simulation will end after amount of generations are reached or the fitness threshold
    winner = p.run(eval_genomes)

    # show final stats
    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
