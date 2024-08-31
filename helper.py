import matplotlib.pyplot as plt
from IPython import display

# Ativa o modo interativo para que as atualizações do gráfico sejam refletidas instantaneamente
plt.ion()

def plot(scores, mean_scores):
    """
    Plota os scores e a média dos scores durante o treinamento.

    Este gráfico é atualizado em tempo real para fornecer uma visualização contínua
    do desempenho do agente conforme ele joga mais jogos.

    Args:
        scores (list): Uma lista com os scores obtidos em cada jogo.
        mean_scores (list): Uma lista com as médias dos scores acumulados até cada jogo.
    """
    # Limpa a saída anterior e prepara para o novo gráfico
    display.clear_output(wait=True)
    display.display(plt.gcf())  # Exibe o gráfico atual
    plt.clf()  # Limpa a figura para preparar o próximo plot

    # Configurações do gráfico
    plt.title('Training...')  # Título do gráfico
    plt.xlabel('Number of Games')  # Rótulo do eixo X
    plt.ylabel('Score')  # Rótulo do eixo Y

    # Plota os scores e a média dos scores
    plt.plot(scores, label='Score')  # Plota os scores individuais
    plt.plot(mean_scores, label='Mean Score')  # Plota a média dos scores

    # Define o limite mínimo do eixo Y para ser 0
    plt.ylim(ymin=0)

    # Adiciona o valor do último score e da última média no gráfico
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Exibe o score do último jogo
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))  # Exibe a média dos scores

    # Exibe o gráfico sem bloquear a execução e pausa por um momento para a atualização
    plt.show(block=False)
    plt.pause(.1)
