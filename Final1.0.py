import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def derivada_manual(vetor):
    """Calcula a derivada manual de um vetor de forma vetorizada."""
    vetor = vetor.astype(np.float32)
    resultado = np.zeros_like(vetor)
    resultado[1:-1] = (vetor[2:] - vetor[:-2]) / 2.0
    resultado[0] = vetor[1] - vetor[0]
    resultado[-1] = vetor[-1] - vetor[-2]
    return resultado

def carregar_imagem(caminho):
    """
    Carrega a imagem. Se for RGB, converte para escala de cinza.
    """
    img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")
    # Se tiver 3 canais, converte para escala de cinza
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def selecionar_perfil(img):
    """Interage com o usuário para selecionar linha ou coluna da imagem."""
    while True:
        tipo = input("Digite 'linha' ou 'coluna' para escolher o perfil: ").strip().lower()
        if tipo not in ['linha', 'coluna']:
            print("Tipo inválido! Digite 'linha' ou 'coluna'.")
            continue
        try:
            indice = int(input(f"Digite o índice da {tipo}: "))
        except ValueError:
            print("Digite um número inteiro válido.")
            continue
        if tipo == 'linha' and 0 <= indice < img.shape[0]:
            return tipo, indice
        elif tipo == 'coluna' and 0 <= indice < img.shape[1]:
            return tipo, indice
        else:
            print(f"Índice fora do intervalo! A imagem tem {img.shape[0]} linhas e {img.shape[1]} colunas.")

def mostrar_imagem_destacada(img, tipo, indice):
    """Mostra a imagem com a linha ou coluna destacada."""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if tipo == 'linha':
        cv2.line(img_color, (0, indice), (img.shape[1]-1, indice), (0, 0, 255), 1)
    elif tipo == 'coluna':
        cv2.line(img_color, (indice, 0), (indice, img.shape[0]-1), (0, 0, 255), 1)

    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title(f'{tipo.capitalize()} {indice} destacada')
    plt.axis('off')
    plt.show()

def estatisticas_brilho(y):
    """Imprime estatísticas básicas do vetor de brilho."""
    print("Estatísticas do brilho:")
    print("Média:", np.mean(y))
    print("Desvio padrão:", np.std(y))
    print("Mínimo:", np.min(y))
    print("Máximo:", np.max(y))
    print("Mediana:", np.median(y))

def interpolacao_manual(x_pontos, y_pontos, x_interp):
    y_interp = []
    for xi in x_interp:
        i = np.searchsorted(x_pontos, xi) - 1
        i = np.clip(i, 0, len(x_pontos) - 2)
        x0, x1 = x_pontos[i], x_pontos[i+1]
        y0, y1 = y_pontos[i], y_pontos[i+1]
        yi = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
        y_interp.append(yi)
    return np.array(y_interp)

def main():
    arquivo = input("Digite o caminho da imagem (RGB ou escala de cinza): ").strip()
    img = carregar_imagem(arquivo)
    print(f"A imagem tem {img.shape[0]} linhas e {img.shape[1]} colunas.")

    tipo, indice = selecionar_perfil(img)

    # Extrair perfil de linha ou coluna
    perfil = img[indice, :] if tipo == 'linha' else img[:, indice]
    y = perfil.astype(np.float32)
    x = np.arange(len(y))

    mostrar_imagem_destacada(img, tipo, indice)
    estatisticas_brilho(y)

    # Histograma
    plt.hist(y, bins=32, color='gray', edgecolor='black')
    plt.title(f'Histograma do Brilho ({tipo} {indice})')
    plt.xlabel('Brilho (0-255)')
    plt.ylabel('Frequência')
    plt.show()

    # Perfil de brilho
    plt.plot(x, y, color='blue')
    plt.title(f'Perfil de Brilho da {tipo.capitalize()} {indice}')
    plt.xlabel('Posição do pixel')
    plt.ylabel('Brilho')
    plt.grid(True)
    plt.show()

    # Derivadas
    dy_numpy = np.gradient(y)
    dy_manual = derivada_manual(y)

    # Comparação
    plt.plot(x, dy_manual, label='Manual', color='green')
    plt.plot(x, dy_numpy, label='NumPy', linestyle='--', color='red')
    plt.legend()
    plt.title('Comparação das Derivadas')
    plt.xlabel('Posição do pixel')
    plt.ylabel('Variação do brilho')
    plt.grid(True)
    plt.show()

    # Interpolação
    x_interp = np.linspace(x[0], x[-1], len(x) * 4)
    func_scipy = interp1d(x, y, kind='linear')
    y_interp_scipy = func_scipy(x_interp)
    y_interp_manual = interpolacao_manual(x, y, x_interp)

    # Comparação quantitativa
    erro = np.abs(y_interp_scipy - y_interp_manual).mean()
    print(f"Erro médio absoluto entre Scipy e Manual: {erro:.6f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o', label='Pontos Originais')
    plt.plot(x_interp, y_interp_scipy, '-', label='Interpolação Scipy')
    plt.plot(x_interp, y_interp_manual, '--', label='Interpolação Manual')
    plt.title('Interpolação Linear do Brilho')
    plt.xlabel('Posição do Pixel')
    plt.ylabel('Brilho (0-255)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()