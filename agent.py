import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Definição de parâmetros
MAX_MEMORY = 100_000  # Tamanho máximo da memória para replay
BATCH_SIZE = 1000     # Tamanho do lote para treinamento
LR = 0.001            # Taxa de aprendizado

class Agent:
    """
    Classe que representa o agente inteligente que jogará o Snake Game.
    Implementa um agente baseado em Deep Q-Learning.
    """

    def __init__(self):
        """
        Inicializa o agente com os parâmetros necessários.
        """
        self.n_games = 0  # Número de jogos jogados
        self.epsilon = 0  # Controle da exploração (randomness)
        self.gamma = 0.9  # Fator de desconto (para Q-learning)
        self.memory = deque(maxlen=MAX_MEMORY)  # Memória para armazenar experiências
        self.model = Linear_QNet(11, 256, 3)  # Rede Neural usada para prever ações
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Treinador do modelo

    def get_state(self, game):
        """
        Extrai o estado atual do jogo em formato binário.

        Args:
            game (SnakeGameAI): Instância do jogo da cobrinha.

        Returns:
            np.array: Vetor representando o estado atual do jogo.
        """
        head = game.snake[0]  # Cabeça da cobra
        point_l = Point(head.x - 20, head.y)  # Ponto à esquerda da cabeça
        point_r = Point(head.x + 20, head.y)  # Ponto à direita da cabeça
        point_u = Point(head.x, head.y - 20)  # Ponto acima da cabeça
        point_d = Point(head.x, head.y + 20)  # Ponto abaixo da cabeça

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Perigo à frente
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Perigo à direita
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direção do movimento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Localização do alimento 
            game.food.x < game.head.x,  # Alimento à esquerda
            game.food.x > game.head.x,  # Alimento à direita
            game.food.y < game.head.y,  # Alimento acima
            game.food.y > game.head.y   # Alimento abaixo
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Armazena uma experiência na memória do agente.

        Args:
            state (np.array): Estado atual.
            action (list): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (np.array): Próximo estado.
            done (bool): Se o jogo terminou.
        """
        self.memory.append((state, action, reward, next_state, done))  # Remove o mais antigo se atingir MAX_MEMORY

    def train_long_memory(self):
        """
        Treina a memória de longo prazo do agente usando amostragem aleatória da memória.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Amostra aleatória da memória
        else:
            mini_sample = self.memory

        # Descompacta as experiências da amostra
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Treina a memória de curto prazo do agente com uma única experiência.

        Args:
            state (np.array): Estado atual.
            action (list): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (np.array): Próximo estado.
            done (bool): Se o jogo terminou.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Decide a próxima ação do agente com base no estado atual.

        Args:
            state (np.array): Estado atual.

        Returns:
            list: Ação a ser tomada (movimento).
        """
        # Exploração vs Exploração
        self.epsilon = 80 - self.n_games  # Diminui à medida que o número de jogos aumenta
        final_move = [0, 0, 0]  # Representação das ações [frente, direita, esquerda]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # Escolha aleatória
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Previsão da rede neural
            move = torch.argmax(prediction).item()  # Ação com maior valor Q
            final_move[move] = 1

        return final_move

def train():
    """
    Função principal para treinar o agente. Executa um loop infinito de jogos, 
    onde o agente joga, treina e armazena suas experiências.
    """
    plot_scores = []        # Armazena as pontuações para plotar
    plot_mean_scores = []   # Armazena as pontuações médias para plotar
    total_score = 0
    record = 0
    agent = Agent()         # Inicializa o agente
    game = SnakeGameAI()    # Inicializa o jogo

    while True:
        # Obtém o estado anterior
        state_old = agent.get_state(game)

        # Obtém a próxima ação do agente
        final_move = agent.get_action(state_old)

        # Executa a ação e obtém o novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Treina a memória de curto prazo
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Armazena a experiência na memória
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Treina a memória de longo prazo e reseta o jogo
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Atualiza os gráficos de pontuação
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
