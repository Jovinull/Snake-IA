# Snake Game AI - Inteligência Artificial para o Jogo da Cobrinha

## Descrição
Este projeto implementa uma **Inteligência Artificial (IA) baseada em Deep Q-Learning** para jogar o clássico jogo da cobrinha (*Snake Game*). A IA é treinada para aprender a jogar de maneira autônoma, maximizando sua pontuação enquanto evita colisões.

## Funcionalidades
- **Aprendizado por Reforço**: Utiliza **Deep Q-Learning** para aprender e melhorar sua performance.
- **Memória de Replay**: Implementa uma **memória longa e curta** para armazenar experiências e aprimorar a tomada de decisões.
- **Treinamento com Rede Neural**: Utiliza **PyTorch** para treinar uma rede neural capaz de prever as melhores ações.
- **Gráficos em Tempo Real**: Exibe gráficos com a evolução da performance da IA ao longo dos jogos.

## 🛠️ Tecnologias Utilizadas
- **Python**
- **PyTorch**
- **NumPy**
- **Matplotlib**
- **Pygame** (para renderizar o jogo)

## Como Executar o Projeto

### Pré-requisitos
- **Python 3.x**
- Instale as dependências necessárias:
  ```sh
  pip install torch numpy pygame matplotlib
  ```

### Passo a Passo
1. Clone o repositório:
   ```sh
   git clone https://github.com/seu-usuario/snake-game-ai.git
   ```
2. Acesse o diretório do projeto:
   ```sh
   cd snake-game-ai
   ```
3. Execute o script para treinar a IA:
   ```sh
   python agent.py
   ```
4. A IA começará a jogar e você poderá visualizar a evolução da sua performance.

## Treinamento e Resultados
Durante o treinamento, a IA aprende a sobreviver mais tempo e comer mais frutas, maximizando a pontuação. O progresso é registrado e pode ser visualizado em gráficos.

## Melhorias Futuras
- Implementação de **Redes Neurais Convolucionais (CNNs)** para melhorar a percepção do ambiente.
- Otimização dos hiperparâmetros do modelo para maior eficiência.
- Testes com **diferentes estratégias de aprendizado** para melhorar a performance da IA.

## Contribuindo
Contribuições são bem-vindas! Para contribuir:
1. Faça um **fork** do projeto.
2. Crie uma **branch** para sua modificação (`git checkout -b minha-modificacao`).
3. Envie suas mudanças (`git push origin minha-modificacao`).
4. Abra um **Pull Request**.

## Licença
Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
