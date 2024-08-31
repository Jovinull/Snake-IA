import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    Rede Neural Linear para o agente de Q-Learning.
    
    Esta rede possui uma camada oculta e é utilizada para prever as ações que o agente deve tomar.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa a rede neural com as camadas especificadas.

        Args:
            input_size (int): Tamanho da camada de entrada (número de features).
            hidden_size (int): Tamanho da camada oculta.
            output_size (int): Tamanho da camada de saída (número de ações possíveis).
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Primeira camada linear
        self.linear2 = nn.Linear(hidden_size, output_size) # Segunda camada linear

    def forward(self, x):
        """
        Realiza o passo forward (propagação para frente) na rede neural.

        Args:
            x (torch.Tensor): Input para a rede neural.

        Returns:
            torch.Tensor: Saída da rede neural.
        """
        x = F.relu(self.linear1(x))  # Aplica ReLU na saída da primeira camada
        x = self.linear2(x)          # Passa pela segunda camada
        return x

    def save(self, file_name='model.pth'):
        """
        Salva o modelo treinado em um arquivo.

        Args:
            file_name (str, optional): Nome do arquivo onde o modelo será salvo. Default é 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)  # Salva os pesos do modelo


class QTrainer:
    """
    Treinador do agente de Q-Learning, responsável por realizar o treinamento do modelo.
    """

    def __init__(self, model, lr, gamma):
        """
        Inicializa o treinador com o modelo, taxa de aprendizado e fator de desconto.

        Args:
            model (Linear_QNet): O modelo de rede neural que será treinado.
            lr (float): Taxa de aprendizado para o otimizador.
            gamma (float): Fator de desconto para o Q-Learning.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Otimizador Adam
        self.criterion = nn.MSELoss()  # Critério de perda (Mean Squared Error)

    def train_step(self, state, action, reward, next_state, done):
        """
        Realiza um passo de treinamento do modelo.

        Args:
            state (np.array): Estado atual.
            action (np.array): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (np.array): Próximo estado.
            done (bool): Indicador se o episódio terminou.
        """
        # Converte entradas para tensores PyTorch
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Se o estado for único, ajusta a forma para (1, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predição dos valores Q com o estado atual
        pred = self.model(state)

        # Clona a predição para ajustar o valor Q alvo
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Calcula a perda e realiza a retropropagação
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
