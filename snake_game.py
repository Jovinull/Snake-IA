import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Inicializa o Pygame
pygame.init()

# Define a fonte para exibir o texto no jogo
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    """
    Enumeração para representar as direções do movimento da cobra no jogo.
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Estrutura para representar um ponto no grid do jogo
Point = namedtuple('Point', 'x, y')

# Definição de cores RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Tamanho do bloco da cobra e velocidade do jogo
BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    """
    Classe que representa o jogo da cobrinha controlado por uma IA.
    """

    def __init__(self, w=640, h=480):
        """
        Inicializa o jogo da cobrinha com uma interface gráfica.

        Args:
            w (int): Largura da tela do jogo.
            h (int): Altura da tela do jogo.
        """
        self.w = w
        self.h = h
        # Inicializa a tela de exibição do jogo
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Reinicia o estado do jogo, incluindo a direção inicial da cobra, 
        sua posição e o alimento.
        """
        self.direction = Direction.RIGHT

        # Define a posição inicial da cobra no centro da tela
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        Coloca o alimento em uma posição aleatória no grid, 
        garantindo que não coincida com a posição da cobra.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Executa um passo do jogo com base na ação fornecida.

        Args:
            action (list): Lista que representa a ação a ser tomada pela IA.

        Returns:
            tuple: Recompensa obtida, se o jogo terminou, e a pontuação atual.
        """
        self.frame_iteration += 1
        
        # 1. Captura as entradas do usuário (eventos)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move a cobra de acordo com a ação fornecida
        self._move(action)  # Atualiza a posição da cabeça da cobra
        self.snake.insert(0, self.head)
        
        # 3. Verifica se o jogo acabou (colisão ou tempo limite)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Coloca um novo alimento ou move a cobra
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Atualiza a interface do usuário e o relógio do jogo
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Retorna se o jogo terminou e a pontuação atual
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Verifica se ocorreu uma colisão com as bordas ou com o corpo da cobra.

        Args:
            pt (Point, opcional): Ponto a ser verificado. Se não fornecido, 
                                  verifica a posição atual da cabeça da cobra.

        Returns:
            bool: True se houver uma colisão, False caso contrário.
        """
        if pt is None:
            pt = self.head
        # Verifica colisão com as bordas da tela
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Verifica colisão com o próprio corpo da cobra
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Atualiza a interface do usuário, desenhando a cobra e o alimento na tela.
        """
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move a cobra para uma nova direção com base na ação fornecida.

        Args:
            action (list): Lista que representa a ação a ser tomada pela IA.
                        [straight, right, left]

        Detalhes:
            A ordem das direções é definida em sentido horário: [direita, baixo, esquerda, cima].

            O módulo 4 é utilizado para garantir que o índice da nova direção da cobra 
            permaneça dentro do intervalo válido (0 a 3). Como há apenas 4 direções possíveis, 
            o módulo 4 evita que o índice ultrapasse esses limites, permitindo que a direção 
            circule corretamente entre as opções.
        """
        # Define a ordem das direções em sentido horário (direita, baixo, esquerda, cima)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # Encontra o índice da direção atual da cobra na lista `clock_wise`
        idx = clock_wise.index(self.direction)

        # Verifica qual ação a IA escolheu e ajusta a direção da cobra com base nisso
        if np.array_equal(action, [1, 0, 0]):
            # Se a ação for [1, 0, 0], a cobra continua na mesma direção (nenhuma mudança)
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Se a ação for [0, 1, 0], a cobra deve virar à direita
            # Calcula o próximo índice em sentido horário (somando 1 e usando módulo 4 para manter dentro dos limites)
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Se a ação for [0, 0, 1], a cobra deve virar à esquerda
            # Calcula o índice anterior em sentido horário (subtraindo 1 e usando módulo 4 para manter dentro dos limites)
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        # Atualiza a posição da cabeça da cobra
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
