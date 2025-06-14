# Snake Game AI - Inteligência Artificial para o Jogo da Cobrinha

## Descrição

Este projeto implementa uma **Inteligência Artificial (IA) baseada em Deep Q-Learning** para jogar o clássico jogo da cobrinha (*Snake Game*). A IA é treinada para aprender a jogar de maneira autônoma, maximizando sua pontuação enquanto evita colisões.

## Funcionalidades

* **Aprendizado por Reforço**: Utiliza **Deep Q-Learning** para aprender e melhorar sua performance.
* **Memória de Replay**: Implementa uma **memória longa e curta** para armazenar experiências e aprimorar a tomada de decisões.
* **Treinamento com Rede Neural**: Utiliza **PyTorch** para treinar uma rede neural capaz de prever as melhores ações.
* **Gráficos em Tempo Real**: Exibe gráficos com a evolução da performance da IA ao longo dos jogos.

## Tecnologias Utilizadas

* **Python**
* **PyTorch**
* **NumPy**
* **Matplotlib**
* **Pygame** (para renderizar o jogo)

## Como Executar o Projeto

### Pré-requisitos

* **Python 3.x instalado** (recomendado >= 3.10)
* Ter o **Git** instalado para clonar o repositório (ou baixar o ZIP manualmente)

### Dependências do Sistema (Obrigatório para Windows)

Para que o **PyTorch** funcione corretamente no Windows, é necessário instalar os **Microsoft Visual C++ Redistributables**. Sem isso, você poderá encontrar erros como:

```
OSError: [WinError 126] Não foi possível encontrar o módulo especificado. Error loading ...\torch\lib\fbgemm.dll
```

**Baixe e instale os redistribuíveis aqui:**

* [Visual C++ Redistributable - Microsoft](https://learn.microsoft.com/pt-br/cpp/windows/latest-supported-vc-redist?view=msvc-170)

Instale o arquivo:

* `vc_redist.x64.exe` (para sistemas 64 bits)

---

### Instalação do PyTorch

O projeto pode ser executado tanto na **CPU** quanto na **GPU** (se você possuir uma placa de vídeo da NVIDIA com CUDA instalado corretamente).

#### Instalar versão CPU (Recomendado para a maioria dos usuários):

Mais simples e sem necessidade de drivers adicionais.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 🔹 Instalar versão GPU (Se você possui CUDA instalado):

Acesse o site oficial para gerar o comando de instalação específico para sua versão de CUDA:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Exemplo (CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Instalação das Dependências

Execute no terminal dentro do diretório do projeto:

```bash
pip install numpy pygame matplotlib
```

### Passo a Passo para Execução

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/snake-game-ai.git
   ```
2. Acesse o diretório do projeto:

   ```bash
   cd snake-game-ai
   ```
3. Execute o script principal:

   ```bash
   python agent.py
   ```
4. A IA começará a jogar e você poderá visualizar a evolução da sua performance.

---

## Treinamento e Resultados

Durante o treinamento, a IA aprende a sobreviver mais tempo e comer mais frutas, maximizando a pontuação. O progresso é registrado e exibido por meio de gráficos em tempo real.

---

## Melhorias Futuras

* Implementação de **Redes Neurais Convolucionais (CNNs)** para melhorar a percepção do ambiente.
* Otimização dos hiperparâmetros do modelo para maior eficiência.
* Testes com **diferentes estratégias de aprendizado** e arquiteturas para melhorar a performance da IA.

---

## Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um **fork** do projeto.
2. Crie uma **branch** para sua modificação:

   ```bash
   git checkout -b minha-modificacao
   ```
3. Envie suas alterações:

   ```bash
   git push origin minha-modificacao
   ```
4. Abra um **Pull Request**.

---

## 📜 Licença

Este projeto está licenciado sob a **MIT License** – consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
