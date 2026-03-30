#!/usr/bin/env python
# -*- coding: utf-8 -*-

# argparse permite receber argumentos pela linha de comando,
# como os caminhos dos arquivos clean, poisoned e saídas.
import argparse

# json é usado para ler o results.json de cada execução
# e também para salvar o summary.json final.
import json

# os é usado para criar diretórios de saída quando necessário.
import os

# pandas é usado para transformar os dados em DataFrame,
# fazer merge entre execuções e calcular colunas delta com facilidade.
import pandas as pd


def load_results_json(path: str) -> pd.DataFrame:
    """
    Lê um arquivo results.json gerado pelo analyze.py e devolve um DataFrame.

    O que faz:
    - abre o JSON
    - recupera a lista "by_round"
    - converte essa lista em DataFrame

    Por que usamos:
    - o analyze.py salva o histórico por round em JSON
    - para comparar duas execuções round a round, o formato tabular é mais prático
    - pandas facilita merge, ordenação e cálculo das diferenças

    Retorno:
    - DataFrame com colunas típicas:
      round, accuracy, precision, recall, f1

    Observação:
    - se o JSON não tiver "by_round", devolvemos um DataFrame vazio
      com as colunas esperadas, para manter robustez no restante do pipeline
    """

    # Abre o arquivo JSON de entrada.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recupera a lista de métricas por round.
    by_round = data.get("by_round", [])

    # Se não houver histórico, devolve DataFrame vazio com colunas esperadas.
    # Isso evita quebra no restante do código.
    if not by_round:
        return pd.DataFrame(columns=["round", "accuracy", "precision", "recall", "f1"])

    # Converte a lista de dicionários em DataFrame.
    return pd.DataFrame(by_round)


def main():
    """
    Função principal do script.

    Fluxo:
    1. lê argumentos da linha de comando
    2. carrega results.json da execução clean
    3. carrega results.json da execução poisoned
    4. renomeia as colunas para separar clean e poison
    5. faz merge por round
    6. calcula deltas (clean - poisoned)
    7. salva CSV combinado
    8. gera summary.json com estatísticas resumidas

    Esse script existe para preparar a base da análise comparativa.
    """

    # Cria parser de argumentos CLI.
    parser = argparse.ArgumentParser(description="Compara duas execuções (clean vs poisoned)")

    # Arquivo results.json da execução clean.
    parser.add_argument("--clean", required=True, help="Caminho para results.json da execução clean")

    # Arquivo results.json da execução poisoned.
    parser.add_argument("--poison", required=True, help="Caminho para results.json da execução poisoned")

    # CSV final que conterá as métricas lado a lado por round.
    parser.add_argument("--out-csv", required=True, help="CSV de saída com comparação round a round")

    # JSON resumo com estatísticas consolidadas.
    parser.add_argument("--out-summary", required=True, help="JSON de resumo")

    # Lê os argumentos fornecidos pelo usuário.
    args = parser.parse_args()

    # Carrega a execução clean e renomeia as colunas métricas.
    # Por que renomear:
    # - para evitar conflito com as colunas da execução poisoned
    # - para deixar explícito qual métrica vem de qual experimento
    df_clean = load_results_json(args.clean).rename(
        columns={
            "accuracy": "accuracy_clean",
            "precision": "precision_clean",
            "recall": "recall_clean",
            "f1": "f1_clean",
        }
    )

    # Faz o mesmo para a execução poisoned.
    df_poison = load_results_json(args.poison).rename(
        columns={
            "accuracy": "accuracy_poison",
            "precision": "precision_poison",
            "recall": "recall_poison",
            "f1": "f1_poison",
        }
    )

    # Faz merge entre os dois DataFrames usando a coluna "round".
    # how="outer" garante que rounds presentes em um e ausentes em outro
    # não sejam perdidos.
    #
    # Exemplo:
    # round | accuracy_clean | accuracy_poison | ...
    df = pd.merge(df_clean, df_poison, on="round", how="outer").sort_values("round").reset_index(drop=True)

    # Calcula os deltas para cada métrica:
    # delta = clean - poisoned
    #
    # Interpretação:
    # - delta > 0  => clean melhor que poisoned
    # - delta < 0  => poisoned melhor que clean
    # - delta = 0  => empate
    for metric in ["accuracy", "precision", "recall", "f1"]:
        c = f"{metric}_clean"
        p = f"{metric}_poison"

    # garante colunas mesmo se faltarem
        if c not in df:
            df[c] = pd.NA
        if p not in df:
            df[p] = pd.NA

        df[f"delta_{metric}"] = df[c] - df[p]

    # Garante que a pasta de saída do CSV exista.
    # os.path.dirname(...) pode retornar string vazia se o arquivo estiver na pasta atual,
    # então usamos "or '.'" como fallback.
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    # Salva o CSV combinado round a round.
    # Esse CSV será usado depois pelo analyze_results.py para gerar gráficos comparativos.
    df.to_csv(args.out_csv, index=False)

    # Cria dicionário de resumo da comparação.
    summary = {
        # Guarda os caminhos dos arquivos usados como entrada.
        "clean_file": args.clean,
        "poison_file": args.poison,

        # Número de rounds distintos encontrados.
        "rounds": int(df["round"].nunique()) if not df.empty else 0,

        # Dicionário onde iremos salvar o resumo por métrica.
        "metrics": {},
    }

    # Para cada métrica, calcula estatísticas resumidas dos deltas.
    for metric in ["accuracy", "precision", "recall", "f1"]:
        delta_col = f"delta_{metric}"

        # Se o DataFrame estiver vazio, não há o que resumir.
        if df.empty:
            summary["metrics"][metric] = {
                "delta_mean": None,
                "delta_final": None,
                "delta_max": None,
            }
        else:
            summary["metrics"][metric] = {
                # Média do delta ao longo de todos os rounds.
                # Dá uma noção do impacto médio do poisoning.
                "delta_mean": float(df[delta_col].mean()),

                # Delta no último round.
                # Muito útil em análise final de convergência.
                "delta_final": float(df[delta_col].iloc[-1]),

                # Maior delta observado.
                # Mostra o maior afastamento clean - poisoned ao longo da execução.
                "delta_max": float(df[delta_col].max()),
            }

    # Salva o resumo em JSON.
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Mensagens finais no terminal para o usuário saber onde os arquivos foram salvos.
    print("Comparação salva em:", args.out_csv)
    print("Resumo salvo em:", args.out_summary)


# Ponto de entrada do script.
# Garante que o main() só rode quando este arquivo for executado diretamente.
if __name__ == "__main__":
    main()