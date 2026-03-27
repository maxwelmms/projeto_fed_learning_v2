#!/usr/bin/env python
# -*- coding: utf-8 -*-

# argparse permite receber argumentos pela linha de comando,
# como o CSV de entrada e o diretório de saída dos gráficos.
import argparse

# json é usado para salvar o summary.json final com estatísticas resumidas.
import json

# os é usado para criar diretórios e montar caminhos de arquivos.
import os

# Define o backend não interativo do matplotlib.
# Isso é importante para gerar imagens em ambiente sem interface gráfica.
os.environ.setdefault("MPLBACKEND", "Agg")

# Importa matplotlib e força backend "Agg".
# Esse backend é adequado para gerar PNGs em scripts executados via terminal.
import matplotlib
matplotlib.use("Agg")

# pyplot é usado para criar e salvar os gráficos.
import matplotlib.pyplot as plt

# pandas é usado para ler o CSV gerado pelo compare_export.py
# e facilitar os cálculos estatísticos e o acesso às colunas.
import pandas as pd


def plot_metric(df, metric, outdir):
    """
    Gera um gráfico comparando uma métrica entre as execuções
    clean e poisoned ao longo dos rounds.

    O que faz:
    - plota a curva da métrica para a execução clean
    - plota a curva da métrica para a execução poisoned
    - salva a figura em PNG

    Por que usamos:
    - permite comparar visualmente a evolução das duas execuções
    - ajuda a ver se o poisoning degradou o desempenho
    - é um gráfico útil para discussão

    Parâmetros:
    - df: DataFrame com colunas da comparação round a round
    - metric: nome da métrica ("accuracy", "precision", "recall", "f1")
    - outdir: diretório onde o gráfico será salvo

    Saída:
    - compare_<metric>.png
    """

    # Cria figura e eixo com tamanho fixo.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Curva da execução clean.
    ax.plot(df["round"], df[f"{metric}_clean"], marker="o", label=f"{metric} clean")

    # Curva da execução poisoned.
    ax.plot(df["round"], df[f"{metric}_poison"], marker="o", label=f"{metric} poisoned")

    # Rótulos e título.
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} — Clean vs Poisoned")

    # Grade para facilitar leitura visual.
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legenda para distinguir as duas curvas.
    ax.legend()

    # Ajuste automático de layout.
    fig.tight_layout()

    # Caminho do arquivo de saída.
    path = os.path.join(outdir, f"compare_{metric}.png")

    # Salva em PNG.
    fig.savefig(path, dpi=160)

    # Fecha a figura para liberar memória.
    plt.close(fig)

    return path


def plot_delta(df, metric, outdir):
    """
    Gera um gráfico da diferença (delta) entre clean e poisoned
    para uma métrica ao longo dos rounds.

    O que faz:
    - plota a coluna delta_<metric>
    - adiciona uma linha horizontal em y=0
    - salva a figura em PNG

    Por que usamos:
    - o gráfico de comparação mostra duas curvas
    - o gráfico de delta mostra diretamente o impacto do poisoning
    - facilita ver magnitude e sinal da diferença:
      * delta > 0  => clean melhor
      * delta < 0  => poisoned melhor
      * delta = 0  => empate

    Parâmetros:
    - df: DataFrame com colunas delta_<metric>
    - metric: nome da métrica
    - outdir: pasta de saída

    Saída:
    - delta_<metric>.png
    """

    # Cria figura e eixo.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Plota a curva do delta ao longo dos rounds.
    ax.plot(df["round"], df[f"delta_{metric}"], marker="o")

    # Linha horizontal em zero.
    # Muito útil para visualizar quando o delta muda de sinal.
    ax.axhline(0.0, linestyle="--")

    # Rótulos e título.
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Δ {metric}")
    ax.set_title(f"Delta {metric} (clean - poisoned)")

    # Grade.
    ax.grid(True, linestyle="--", alpha=0.6)

    # Ajuste de layout.
    fig.tight_layout()

    # Caminho final do arquivo.
    path = os.path.join(outdir, f"delta_{metric}.png")

    # Salva em PNG.
    fig.savefig(path, dpi=160)

    # Fecha para liberar memória.
    plt.close(fig)

    return path


def main():
    """
    Função principal do script.

    Fluxo:
    1. lê argumentos da linha de comando
    2. cria diretório de saída
    3. lê o CSV combinado gerado por compare_export.py
    4. gera gráficos clean vs poisoned para cada métrica
    5. gera gráficos delta para cada métrica
    6. calcula estatísticas resumidas
    7. salva summary.json
    8. imprime no terminal os arquivos gerados

    Esse script transforma a comparação tabular em visualização e resumo.
    """

    # Cria parser de argumentos.
    parser = argparse.ArgumentParser(description="Analisa CSV de comparação clean vs poisoned")

    # CSV gerado pelo compare_export.py
    parser.add_argument("--csv", required=True, help="CSV gerado por compare_export.py")

    # Diretório de saída dos gráficos e do summary.
    parser.add_argument("--outdir", required=True, help="Diretório de saída")

    # Lê argumentos da CLI.
    args = parser.parse_args()

    # Garante que o diretório de saída exista.
    os.makedirs(args.outdir, exist_ok=True)

    # Lê o CSV em DataFrame.
    df = pd.read_csv(args.csv)

    # Lista dos caminhos dos arquivos gerados.
    paths = []

    # Estrutura do resumo final.
    summary = {
        "input_csv": args.csv,
        "metrics": {},
    }

    # Para cada métrica, gera:
    # - gráfico comparativo clean vs poisoned
    # - gráfico delta
    # - estatísticas resumo
    for metric in ["accuracy", "precision", "recall", "f1"]:
        # Gráfico das duas execuções.
        paths.append(plot_metric(df, metric, args.outdir))

        # Gráfico do delta.
        paths.append(plot_delta(df, metric, args.outdir))

        # Estatísticas resumidas da métrica.
        summary["metrics"][metric] = {
            # Média do delta ao longo dos rounds.
            "delta_mean": float(df[f"delta_{metric}"].mean()),

            # Delta no último round.
            "delta_final": float(df[f"delta_{metric}"].iloc[-1]),

            # Maior delta observado.
            "delta_max": float(df[f"delta_{metric}"].max()),
        }

    # Caminho do JSON resumo.
    summary_path = os.path.join(args.outdir, "summary.json")

    # Salva o resumo.
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Mostra no terminal tudo o que foi gerado.
    print("Arquivos gerados:")
    for p in paths:
        print(" -", p)
    print(" -", summary_path)


# Ponto de entrada do script.
# Garante que main() só seja executado quando o arquivo for chamado diretamente.
if __name__ == "__main__":
    main()