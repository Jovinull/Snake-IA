import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point

# Constantes para o tamanho máximo da memória, tamanho do lote e taxa de aprendizado
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """
    Classe que representa o agente de aprendizado de máquina para jogar Snake.
    Utiliza uma rede neural para tomar decisões com base no estado atual do jogo.
    """

    def __init__(self):
        """
        Inicializa o agente com os parâmetros necessários, como contagem de jogos,
        taxa de exploração (epsilon), taxa de desconto (gamma), memória de experiências,
        modelo de rede neural e treinador.
        """
        self.n_games = 0  # Contador de jogos jogados
        self.epsilon = 0  # Controle de aleatoriedade nas ações
        self.gamma = 0    # Taxa de desconto para recompensas futuras
        self.memory = deque(maxlen=MAX_MEMORY)  # Memória para armazenar experiências passadas
        self.model = None  # Placeholder para o modelo de rede neural (a ser implementado)
        self.trainer = None  # Placeholder para o treinador do modelo (a ser implementado)

    def get_state(self, snake_game):
        """
        Obtém o estado atual do jogo, que será usado pelo agente para tomar decisões.
        
        Args:
            snake_game (SnakeGameAI): Instância atual do jogo da cobrinha.
        
        Returns:
            np.array: Array representando o estado atual do jogo.
        """
        head = snake_game.snake[0]  # Cabeça da cobra
        point_l = Point(head.x - 20, head.y)  # Ponto à esquerda da cabeça
        point_r = Point(head.x + 20, head.y)  # Ponto à direita da cabeça
        point_u = Point(head.x, head.y - 20)  # Ponto acima da cabeça
        point_d = Point(head.x, head.y + 20)  # Ponto abaixo da cabeça

        # Direção atual da cobra
        dir_l = snake_game.direction == Direction.LEFT
        dir_r = snake_game.direction == Direction.RIGHT
        dir_u = snake_game.direction == Direction.UP
        dir_d = snake_game.direction == Direction.DOWN

        # Construção do estado com base nas condições de perigo, direção e localização do alimento
        state = [
            # Perigo à frente
            (dir_r and snake_game.is_collision(point_r)) or 
            (dir_l and snake_game.is_collision(point_l)) or 
            (dir_u and snake_game.is_collision(point_u)) or 
            (dir_d and snake_game.is_collision(point_d)),

            # Perigo à direita
            (dir_u and snake_game.is_collision(point_r)) or 
            (dir_d and snake_game.is_collision(point_l)) or 
            (dir_l and snake_game.is_collision(point_u)) or 
            (dir_r and snake_game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_d and snake_game.is_collision(point_r)) or 
            (dir_u and snake_game.is_collision(point_l)) or 
            (dir_r and snake_game.is_collision(point_u)) or 
            (dir_l and snake_game.is_collision(point_d)),
            
            # Direção atual da cobra
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Localização do alimento em relação à cabeça da cobra
            snake_game.food.x < snake_game.head.x,  # Alimento à esquerda
            snake_game.food.x > snake_game.head.x,  # Alimento à direita
            snake_game.food.y < snake_game.head.y,  # Alimento acima
            snake_game.food.y > snake_game.head.y   # Alimento abaixo
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Armazena uma experiência na memória do agente.
        
        Args:
            state (np.array): Estado atual do jogo.
            action (list): Ação tomada pelo agente.
            reward (int): Recompensa obtida após a ação.
            next_state (np.array): Próximo estado do jogo após a ação.
            done (bool): Indica se o jogo terminou após a ação.
        """
        self.memory.append((state, action, reward, next_state, done))  # Remove o item mais antigo se a memória estiver cheia

    def train_long_memory(self):
        """
        Treina o agente usando amostras da memória de experiências passadas.
        Utiliza um método de replay para melhorar a eficiência do treinamento.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Seleciona um lote aleatório de experiências
        else:
            mini_sample = self.memory  # Usa toda a memória disponível se for menor que o tamanho do lote

        # Descompacta as experiências em estados, ações, recompensas, próximos estados e término
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Treina o agente usando uma única experiência (memória curta).
        
        Args:
            state (np.array): Estado atual do jogo.
            action (list): Ação tomada pelo agente.
            reward (int): Recompensa obtida após a ação.
            next_state (np.array): Próximo estado do jogo após a ação.
            done (bool): Indica se o jogo terminou após a ação.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Decide a próxima ação do agente com base no estado atual.
        Equilibra exploração e exploração usando a estratégia epsilon-greedy.
        
        Args:
            state (np.array): Estado atual do jogo.
        
        Returns:
            list: Ação a ser executada no jogo, representada como uma lista binária.
        """
        # Define a taxa de exploração com base no número de jogos jogados
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]  # [seguir em frente, virar à direita, virar à esquerda]

        if random.randint(0, 200) < self.epsilon:
            # Escolhe uma ação aleatória (exploração)
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Escolhe a melhor ação baseada no modelo (exploração)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Previsão do modelo para o estado atual
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    """
    Função principal para treinar o agente.
    Executa jogos contínuos, treinando o agente com experiências passadas e atualizando o modelo.
    """
    plot_scores = []          # Lista para armazenar pontuações de cada jogo
    plot_mean_scores = []     # Lista para armazenar a média das pontuações
    total_score = 0           # Pontuação total acumulada
    record = 0                # Melhor pontuação alcançada até o momento
    agent = Agent()           # Instancia o agente
    game = SnakeGameAI()      # Instancia o jogo da cobrinha

    while True:
        # Obtém o estado atual do jogo
        state_old = agent.get_state(game)

        # O agente decide qual ação tomar
        final_move = agent.get_action(state_old)

        # Executa a ação no jogo e obtém a recompensa, estado final e se o jogo terminou
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Treina a memória curta com a experiência recente
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Armazena a experiência na memória do agente
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Reinicia o jogo após o término
            game.reset()
            agent.n_games += 1

            # Treina a memória longa com as experiências armazenadas
            agent.train_long_memory()

            # Atualiza o recorde se a pontuação atual for maior
            if score > record:
                record = score
                # agent.model.save()  # Salva o modelo treinado (implementação necessária)

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            # TODO: Implementar plotagem das pontuações

if __name__ == '__main__':
    train()
