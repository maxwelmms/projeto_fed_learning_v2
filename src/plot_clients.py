#!/usr/bin/env python
# -*- coding: utf-8 -*-

# argparse é usado para receber argumentos pela linha de comando,
# como o caminho do results_clients.json e o diretório de saída dos plots.
import argparse

# json é usado para ler o arquivo results_clients.json,
# gerado pelo analyze.py ao final da simulação.
import json

# os é usado para manipular caminhos e criar diretórios.
import os

# sys é usado aqui para encerrar o programa após mostrar a ajuda.
import sys

# NumPy é usado principalmente para representar valores ausentes (np.nan)
# e para facilitar consistência numérica.
import numpy as np

# pandas é usado para transformar os dados dos clientes em DataFrame,
# o que facilita agrupamentos, ordenação, pivot e exportação CSV.
import pandas as pd

# Define backend não interativo para o matplotlib.
# Isso é importante para gerar figuras em PNG sem precisar de interface gráfica.
os.environ.setdefault("MPLBACKEND", "Agg")

# Importa matplotlib e força backend "Agg".
# Esse backend é ideal para execução em terminal/servidor.
import matplotlib
matplotlib.use("Agg")

# pyplot é a interface usada para criar os gráficos.
import matplotlib.pyplot as plt


def load_clients_json(path):
    """
    Lê o arquivo results_clients.json e transforma o conteúdo em um DataFrame.

    O que faz:
    - abre o JSON com as métricas locais dos clientes
    - percorre cliente por cliente
    - percorre round por round de cada cliente
    - transforma tudo em uma estrutura tabular (lista de dicionários)
    - converte essa lista em pandas.DataFrame

    Por que usamos:
    - o analyze.py salva métricas locais em JSON hierárquico
    - para gerar gráficos, o formato tabular é muito mais prático
    - o DataFrame facilita pivot_table, groupby, sort_values etc.

    Retorna:
    - df: DataFrame com métricas por cliente e por round
    - data: JSON original completo, caso precisemos consultar metadados depois
    """

    # Abre o arquivo JSON.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recupera o dicionário de clientes.
    # Espera-se uma estrutura do tipo:
    # {
    #   "0": [ {round:1,...}, {round:2,...} ],
    #   "1": [ ... ],
    #   ...
    # }
    clients = data.get("clients", {})

    # Lista de linhas tabulares.
    rows = []

    # Percorre cada cliente e sua lista de métricas.
    for cid_str, metrics_list in clients.items():
        for m in metrics_list:
            rows.append(
                {
                    # Id do cliente convertido para inteiro.
                    "client": int(cid_str),

                    # Round atual. Se não existir, usa -1 como fallback.
                    "round": int(m.get("round", -1)),

                    # Métricas locais.
                    # Se não existirem, usa NaN para manter a consistência tabular.
                    "accuracy": float(m.get("accuracy", np.nan)),
                    "precision": float(m.get("precision", np.nan)),
                    "recall": float(m.get("recall", np.nan)),
                    "f1": float(m.get("f1", np.nan)),
                    "poisoned": bool(m.get("poisoned", False)),
                    "poison_rate": float(m.get("poison_rate", 0.0)),

                    # Tamanhos locais de treino e teste.
                    # Isso é útil para auditoria e inspeção de heterogeneidade.
                    "n_train_local": int(m.get("n_train_local", 0)),
                    "n_test_local": int(m.get("n_test_local", 0)),
                }
            )

    # Retorna DataFrame com todas as linhas e também o JSON bruto.
    return pd.DataFrame(rows), data


def ensure_outdir(outdir):
    """
    Garante que o diretório de saída exista.

    Por que usamos:
    - antes de salvar PNGs ou CSVs, precisamos ter certeza
      de que a pasta de saída já existe
    - exist_ok=True evita erro se a pasta já estiver criada
    """
    os.makedirs(outdir, exist_ok=True)


def plot_heatmap_accuracy(df, outdir):
    """
    Gera um heatmap de acurácia por cliente x round.

    O que faz:
    - reorganiza o DataFrame em formato matricial:
      linhas = clientes
      colunas = rounds
      valores = accuracy
    - plota essa matriz como imagem (heatmap)

    Por que usamos:
    - esse gráfico mostra rapidamente quais clientes tiveram melhor/pior desempenho
      ao longo dos rounds
    - é muito útil para visualizar heterogeneidade entre clientes
    - em FL, clientes diferentes podem aprender em ritmos diferentes

    Saída:
    - clients_heatmap_accuracy.png
    """

    # Pivot da tabela:
    # índice = cliente
    # colunas = round
    # valor = accuracy média
    if df.empty:
        return None
    
    pvt = df.pivot_table(index="client", columns="round", values="accuracy", aggfunc="mean")

    # Tamanho da figura:
    # largura fixa e altura adaptada ao número de clientes.
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(pvt.index))))

    # Exibe a matriz como imagem.
    # aspect="auto" ajusta proporções automaticamente.
    # interpolation="nearest" evita suavização da imagem.
    im = ax.imshow(pvt.values, aspect="auto", interpolation="nearest")

    # Define ticks do eixo X com os rounds.
    ax.set_xticks(range(len(pvt.columns)))
    ax.set_xticklabels([str(c) for c in pvt.columns])

    # Define ticks do eixo Y com os ids dos clientes.
    ax.set_yticks(range(len(pvt.index)))
    ax.set_yticklabels([str(i) for i in pvt.index])

    # Rótulos dos eixos e título.
    ax.set_xlabel("Round")
    ax.set_ylabel("Cliente")
    ax.set_title("Heatmap — Acurácia por Cliente x Round")

    # Adiciona barra de cores para indicar magnitude da acurácia.
    fig.colorbar(im, ax=ax)

    # Ajusta layout para evitar sobreposição.
    fig.tight_layout()

    # Caminho final do arquivo.
    path = os.path.join(outdir, "clients_heatmap_accuracy.png")

    # Salva em PNG.
    fig.savefig(path, dpi=160)

    # Fecha a figura para liberar memória.
    plt.close(fig)

    return path


def plot_mean_curves(df, outdir):
    """
    Gera curvas da média das métricas locais dos clientes por round.

    O que faz:
    - agrupa o DataFrame por round
    - calcula a média entre clientes para accuracy, precision, recall e f1
    - gera um gráfico separado para cada métrica

    Por que usamos:
    - complementa os plots globais do servidor
    - mostra a tendência média do comportamento local dos clientes
    - ajuda a comparar se os clientes, em média, estão melhorando ou não

    Saídas:
    - clients_mean_curve_accuracy.png
    - clients_mean_curve_precision.png
    - clients_mean_curve_recall.png
    - clients_mean_curve_f1.png
    """

    # Agrupa por round e calcula média das métricas entre todos os clientes.
    g = df.groupby("round", as_index=False)[["accuracy", "precision", "recall", "f1"]].mean()

    # Lista de caminhos gerados.
    paths = []

    # Gera um gráfico por métrica.
    for metric in ["accuracy", "precision", "recall", "f1"]:
        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Curva da métrica média por round.
        ax.plot(g["round"], g[metric], marker="o")

        ax.set_xlabel("Round")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} média dos clientes por round")
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()

        # Nome do arquivo de saída.
        path = os.path.join(outdir, f"clients_mean_curve_{metric}.png")

        # Salva a figura.
        fig.savefig(path, dpi=160)
        plt.close(fig)

        paths.append(path)

    return paths


def plot_final_bars(df, outdir):
    """
    Gera gráfico de barras com a acurácia final de cada cliente.

    O que faz:
    - ordena o DataFrame por cliente e round
    - pega o último registro de cada cliente
    - plota a accuracy final por cliente

    Por que usamos:
    - permite comparar o desempenho final entre clientes
    - útil para mostrar heterogeneidade residual ao final do treinamento
    - é um gráfico simples e direto

    Saída:
    - clients_final_accuracy_bars.png
    """

    # Ordena e pega a última linha de cada cliente.
    last = df.sort_values(["client", "round"]).groupby("client").tail(1)

    # Ajusta largura da figura ao número de clientes.
    fig, ax = plt.subplots(figsize=(max(8, int(0.6 * max(1, len(last)))), 4.5))

    # Gráfico de barras:
    # eixo X = id do cliente
    # eixo Y = accuracy final
    ax.bar(last["client"].astype(str).values, last["accuracy"].values)

    ax.set_xlabel("Cliente")
    ax.set_ylabel("Acurácia (round final do cliente)")
    ax.set_title("Acurácia por cliente no round final")

    # Grid no eixo Y para facilitar comparação visual.
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    # Caminho final da figura.
    path = os.path.join(outdir, "clients_final_accuracy_bars.png")

    # Salva em PNG.
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def show_help_message():
    """
    Exibe uma mensagem curta de uso correto do script e encerra.

    Por que usamos:
    - argparse foi configurado com add_help=False
    - então controlamos manualmente a ajuda
    - isso nos dá uma mensagem de uso mais enxuta e direta
    """
    print("\n[USO CORRETO]")
    print("  python plot_clients.py --in results_clients.json --outdir client_plots\n")
    sys.exit(0)


def main():
    """
    Função principal do script.

    Fluxo:
    1. lê argumentos da linha de comando
    2. valida entrada
    3. cria diretório de saída
    4. carrega results_clients.json em DataFrame
    5. salva versão tabular em CSV
    6. gera plots por cliente
    7. imprime caminhos dos arquivos gerados
    """

    # Cria parser.
    # add_help=False porque usamos uma ajuda personalizada.
    ap = argparse.ArgumentParser(add_help=False)

    # Arquivo de entrada principal: results_clients.json
    ap.add_argument("--in", dest="inp", type=str, help="Caminho do results_clients.json")

    # Diretório de saída dos gráficos.
    ap.add_argument("--outdir", type=str, default="client_plots", help="Diretório de saída dos gráficos")

    # Help manual.
    ap.add_argument("--help", action="store_true", help="Exibe a ajuda")

    # Lê argumentos.
    args = ap.parse_args()

    # Se pediu ajuda ou não informou arquivo de entrada, mostra mensagem e encerra.
    if args.help or not args.inp:
        show_help_message()

    # Verifica se o arquivo realmente existe.
    if not os.path.exists(args.inp):
        print(f"\n[ERRO] Arquivo não encontrado: {args.inp}")
        show_help_message()

    # Garante diretório de saída.
    ensure_outdir(args.outdir)

    # Carrega JSON e transforma em DataFrame tabular.
    df, _ = load_clients_json(args.inp)

    # Caminho do CSV tabular com métricas longas.
    csv_long = os.path.join(args.outdir, "clients_metrics_long.csv")

    # Se houver dados, salva ordenado por cliente e round.
    if not df.empty:
        df.sort_values(["client", "round"]).to_csv(csv_long, index=False)
    else:
        # Se não houver dados, ainda assim salva CSV vazio.
        pd.DataFrame().to_csv(csv_long, index=False)

    # Lista para armazenar os caminhos dos gráficos gerados.
    paths = []

    # Só gera gráficos se houver dados
    if df.empty:
        print("[WARN] Nenhum dado de cliente disponível")
    else:
        paths.append(plot_heatmap_accuracy(df, args.outdir))
        paths.extend(plot_mean_curves(df, args.outdir))
        paths.append(plot_final_bars(df, args.outdir))

    # Exibe no terminal os arquivos gerados.
    print("\nGráficos gerados:")
    for p in paths:
        print(" -", p)

    print("CSV longo:", csv_long)


# Ponto de entrada do script.
# main() só é executado quando esse arquivo é chamado diretamente.
if __name__ == "__main__":
    main()