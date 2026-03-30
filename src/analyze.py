#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Biblioteca para ler argumentos de linha de comando, como:
# --client-csvs, --rounds, --seed etc.
import argparse

# Biblioteca para exportar métricas tabulares em CSV.
import csv

# Usada para gerar timestamps automáticos no run_id quando o usuário não informa um.
import datetime

# Usada para salvar estruturas de dados em JSON (results.json, results_clients.json, run_summary.json).
import json

# Sistema de logs para acompanhar o andamento da execução no terminal.
import logging

# Manipulação de caminhos, diretórios e variáveis de ambiente.
import os

# Medição de tempo total de execução.
import time

# Tipagem estática opcional para melhorar clareza e manutenção do código.
from typing import Dict, List, Optional, Tuple

# Define o backend do matplotlib como "Agg" antes de importar pyplot.
# Isso é importante para gerar gráficos em ambiente sem interface gráfica.
os.environ.setdefault("MPLBACKEND", "Agg")

# Biblioteca base de plot.
import matplotlib

# Força backend não interativo.
matplotlib.use("Agg")

# Interface de plotagem usada para gerar os PNGs de métricas.
import matplotlib.pyplot as plt

# NumPy é usado para vetores, matrizes, classes, concatenação, bootstrap etc.
import numpy as np

# Métricas de avaliação do modelo.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Importa o bundle de dados já pré-processado e a função que monta os datasets federados.
from dataset import FederatedDataBundle, prepare_federated_data

# Importa a função que cria o modelo base de ML.
from model import create_model

# Configuração global de logging.
# Define nível INFO e o formato exibido no terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Logger nomeado deste módulo.
log = logging.getLogger("analyze")

# Diretório atual da execução.
# É global porque várias funções usam esse caminho para salvar arquivos.
run_dir: Optional[str] = None


def compute_metrics(model, X_test, y_test):
    """
    Calcula as métricas de classificação do modelo sobre um conjunto de teste.

    O que faz:
    - usa o modelo para prever os rótulos
    - calcula accuracy, precision, recall e f1

    Por que usamos:
    - precisamos medir o desempenho global e local a cada round
    - isso alimenta os gráficos, JSONs e análises posteriores
    """

    # Gera as previsões do modelo para o conjunto de teste.
    preds = model.predict(X_test)

    # Accuracy: proporção total de acertos.
    acc = accuracy_score(y_test, preds)

    # Precision ponderada pelo suporte das classes.
    # average="weighted" evita tratar todas as classes como igualmente frequentes.
    # zero_division=0 evita erro se alguma classe não tiver predição positiva.
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)

    # Recall ponderado.
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)

    # F1-score ponderado.
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    # Retorna tudo em tupla.
    return acc, prec, rec, f1


def _ensure_model_initialized(model, X_boot, y_boot, all_classes):
    """
    Garante que o modelo esteja inicializado antes de setar coeficientes manualmente.

    Por que isso é necessário:
    - alguns modelos do scikit-learn só criam atributos como:
      coef_, intercept_, classes_
      depois de um fit/partial_fit inicial
    - como no FL recebemos parâmetros do servidor e os injetamos manualmente,
      o modelo precisa existir estruturalmente antes

    Estratégia:
    - tenta acessar coef_, intercept_ e classes_
    - se não existir, faz um partial_fit com um bootstrap mínimo
    """

    try:
        _ = model.coef_
        _ = model.intercept_
        _ = model.classes_
    except Exception:
        # Usa partial_fit porque o modelo escolhido trabalha bem de forma incremental.
        # Também passamos classes=all_classes para garantir shape consistente entre clientes.
        model.partial_fit(X_boot, y_boot, classes=all_classes)


def get_model_params(model) -> List[np.ndarray]:
    """
    Extrai os parâmetros treináveis do modelo.

    O que retorna:
    - coef_      -> matriz de pesos
    - intercept_ -> viés

    Por que usamos:
    - no Flower, cada cliente devolve parâmetros ao servidor
    - o servidor agrega esses parâmetros via FedAvg
    """

    return [model.coef_.copy(), model.intercept_.copy()]


def set_model_params(model, params: List[np.ndarray], X_boot, y_boot, all_classes):
    """
    Injeta parâmetros no modelo.

    O que faz:
    - garante que o modelo já foi inicializado
    - recebe coef e intercept vindos do servidor
    - sobrescreve os parâmetros locais do cliente

    Por que usamos:
    - em FL, a cada round o cliente recebe o modelo global atualizado
    - antes de treinar localmente, ele precisa carregar esses parâmetros
    """

    _ensure_model_initialized(model, X_boot, y_boot, all_classes)

    coef, intercept = params

    # Faz cópia explícita em ndarray para garantir consistência de tipo e isolamento de memória.
    model.coef_ = np.array(coef, copy=True)
    model.intercept_ = np.array(intercept, copy=True)


def save_json_per_round(output_json: str, dataset_desc: List[str], setup: dict, history: Dict[str, List[float]]):
    """
    Salva o histórico global round a round em JSON.

    Estrutura salva:
    - datasets usados
    - configuração da execução
    - métricas por round
    - resumo final

    Por que usamos:
    - esse arquivo é a base para compare_export.py
    - também permite auditoria e reuso posterior sem rerodar experimento
    """

    # Lista de rounds já avaliados.
    rounds = history.get("round", [])

    # Lista que vai armazenar um dicionário por round.
    by_round = []

    # Constrói a lista de métricas por round.
    for i in range(len(rounds)):
        by_round.append(
            {
                "round": int(history["round"][i]),
                "accuracy": float(history["accuracy"][i]),
                "precision": float(history["precision"][i]),
                "recall": float(history["recall"][i]),
                "f1": float(history["f1"][i]),
            }
        )

    # Usa o último round como resumo final.
    final = by_round[-1] if by_round else {}

    # Payload completo do arquivo.
    payload = {
        "datasets": dataset_desc,
        "setup": setup,
        "by_round": by_round,
        "final": final,
    }

    # Escreve em disco.
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_metrics(history: Dict[str, List[float]]):
    """
    Gera os plots globais de métricas por round.

    Saída:
    - trend_accuracy.png
    - trend_precision.png
    - trend_recall.png
    - trend_f1.png

    Por que usamos:
    - para visualizar a evolução do treinamento federado
    - útil para inspeção do comportamento do modelo
    """

    assert run_dir is not None

    # Garante diretório de saída.
    os.makedirs(os.path.join(run_dir, "docs"), exist_ok=True)

    # Eixo X comum: rounds.
    rounds = history["round"]

    # Gera um gráfico por métrica.
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = history[metric]

        plt.figure(figsize=(8, 4))
        plt.plot(rounds, values, marker="o")
        plt.title(f"{metric.capitalize()} por round")
        plt.xlabel("Round")
        plt.ylabel(metric.capitalize())
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Salva como PNG.
        plt.savefig(os.path.join(run_dir, "docs", f"trend_{metric}.png"), dpi=160)
        plt.close()


def aggregate_client_metrics(client_metrics_dir: str, out_json: str, audit_csv: str, dataset_desc: List[str], setup: dict):
    """
    Consolida as métricas locais dos clientes em dois formatos:
    - JSON agregado
    - CSV tabular de auditoria

    Por que usamos:
    - results_clients.json serve para plot_clients.py
    - audit_clients.csv facilita inspeção em planilha/pandas
    """

    # Dicionário com métricas agrupadas por cliente.
    clients_agg = {}

    # Linhas tabulares para o CSV.
    rows = []

    # Percorre todos os JSONs de cliente salvos durante a simulação.
    for fname in sorted(os.listdir(client_metrics_dir)):
        if not (fname.startswith("client_") and fname.endswith(".json")):
            continue

        path = os.path.join(client_metrics_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Salva as métricas daquele cliente no JSON agregado final.
        clients_agg[data["cid"]] = data["metrics"]

        # Converte também para linhas tabulares.
        for m in data["metrics"]:
            rows.append(
                {
                    "client": data["cid"],
                    "client_name": data.get("client_name", ""),
                    "round": m.get("round"),
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "n_train_local": m.get("n_train_local"),
                    "n_test_local": m.get("n_test_local"),
                }
            )

    # Salva o JSON agregado.
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "datasets": dataset_desc,
                "clients": clients_agg,
                "setup": setup,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Se houver linhas, salva também o CSV.
    if rows:
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "client",
                    "client_name",
                    "round",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "n_train_local",
                    "n_test_local",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def run_federated_simulation(
    data: FederatedDataBundle,
    output_json: str,
    num_rounds: int,
    fraction_fit: float,
    seed: int,
    local_epochs: int = 1,
    client_num_cpus: float = 1.0,
    client_num_gpus: float = 0.0,
    ray_init_num_cpus: Optional[int] = None,
    ray_init_num_gpus: Optional[float] = None,
) -> Tuple[Dict[str, List[float]], dict]:
    """
    Função principal da simulação federada.

    Responsabilidades:
    - preparar bootstrap mínimo do modelo
    - definir client_fn
    - definir avaliação global do servidor
    - configurar estratégia FedAvg
    - iniciar simulação Flower
    - exportar resultados

    Essa é a função central do pipeline.
    """

    assert run_dir is not None, "run_dir não inicializado"

    # Conjunto de avaliação global.
    # Pode ser um CSV externo de teste global ou a concatenação dos holdouts locais.
    X_global_eval, y_global_eval = data.global_eval

    # Cria vetor de classes globais [0, 1, 2, ..., K-1].
    # Isso é essencial para manter o mesmo espaço de classes entre todos os clientes.
    all_classes = np.arange(len(data.classes), dtype=np.int64)

    # Bootstrap mínimo com 1 amostra por classe.
    # Serve para inicializar corretamente o modelo incremental.
    X_anchor_blocks = []
    y_anchor_blocks = []

    for cls in all_classes:
        found = False
        for X_train, y_train in data.client_train:
            idx = np.where(y_train == cls)[0]
            if len(idx) > 0:
                X_anchor_blocks.append(X_train[idx[:1]])
                y_anchor_blocks.append(y_train[idx[:1]])
                found = True
                break
        if not found:
            raise ValueError(f"Não foi possível criar bootstrap para a classe {cls}")

    # Junta as amostras bootstrap em uma matriz mínima.
    X_boot = np.vstack(X_anchor_blocks)
    y_boot = np.concatenate(y_anchor_blocks)

    # Histórico global round a round.
    history = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    # Pasta onde cada cliente vai salvar seu JSON local.
    os.makedirs(os.path.join(run_dir, "client_metrics"), exist_ok=True)

    def client_fn(context):
        """
        Factory de clientes Flower.

        O Flower chama essa função para criar cada cliente virtual.
        Aqui vinculamos cada cliente ao seu dataset local.
        """
        import flwr as fl

        # Recupera o id da partição/cliente.
        cid_int = int(context.node_config["partition-id"])

        # Nome do dataset associado a esse cliente.
        client_name = data.client_names[cid_int]

        # Dados de treino local do cliente.
        X_tr_local, y_tr_local = data.client_train[cid_int]

        # Dados de avaliação local do cliente.
        X_te_local, y_te_local = data.client_eval[cid_int]

        # Cria modelo local com seed dependente do cliente.
        model = create_model(random_state=seed + cid_int)

        # Arquivo JSON local para armazenar métricas do cliente.
        client_file = os.path.join(run_dir, "client_metrics", f"client_{cid_int}.json")

        # Se ainda não existir, cria estrutura inicial.
        if not os.path.exists(client_file):
            with open(client_file, "w", encoding="utf-8") as cf:
                json.dump(
                    {"cid": str(cid_int), "client_name": client_name, "metrics": []},
                    cf,
                    ensure_ascii=False,
                )

        class Client(fl.client.NumPyClient):
            """
            Cliente FL no formato NumPyClient.

            Implementa:
            - get_parameters
            - fit
            - evaluate
            """

            def get_parameters(self, config):
                """
                Retorna os parâmetros atuais do modelo local.

                O Flower usa isso para obter parâmetros iniciais ou atualizados.
                """
                _ensure_model_initialized(model, X_boot, y_boot, all_classes)
                return get_model_params(model)

            #---------------------------------#
            def fit(self, parameters, config):
                set_model_params(model, parameters, X_boot, y_boot, all_classes)

                # Treino local do cliente antes da agregação (1 ou mais épocas locais por round)
                for epoch in range(local_epochs):
                    if epoch == 0:
                        model.partial_fit(X_tr_local, y_tr_local, classes=all_classes)
                    else:
                        model.partial_fit(X_tr_local, y_tr_local)

                # Calcula a métrica LOCAL do cliente antes da agregação
                fit_metrics = {}
                if X_te_local.size > 0:
                    acc, prec, rec, f1 = compute_metrics(model, X_te_local, y_te_local)

                    # Salva no JSON do cliente
                    with open(client_file, "r", encoding="utf-8") as cf:
                        payload = json.load(cf)

                    payload["metrics"].append(
                        {
                            "round": int(config.get("server_round", -1)),
                            "accuracy": float(acc),
                            "precision": float(prec),
                            "recall": float(rec),
                            "f1": float(f1),
                            "n_train_local": int(len(X_tr_local)),
                            "n_test_local": int(len(X_te_local)),
                        }
                    )

                    with open(client_file, "w", encoding="utf-8") as cf:
                        json.dump(payload, cf, indent=2, ensure_ascii=False)

                    fit_metrics = {
                        "accuracy": float(acc),
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                    }

                return get_model_params(model), len(X_tr_local), fit_metrics
            #---------------------------------#

            def evaluate(self, parameters, config):
                set_model_params(model, parameters, X_boot, y_boot, all_classes)

                if X_te_local.size == 0:
                    return float(1.0), 0, {}

                acc, prec, rec, f1 = compute_metrics(model, X_te_local, y_te_local)

                return float(1.0 - acc), len(X_te_local), {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }
            
        # Flower 1.20 espera Client, então convertemos NumPyClient com to_client().
        return Client().to_client()

    import flwr as fl

    def get_evaluate_fn():
        """
        Cria a função de avaliação global do servidor.

        O servidor usa essa função a cada round para medir o modelo agregado
        no conjunto global de avaliação.
        """

        def evaluate(server_round: int, parameters: List[np.ndarray], config):
            # Modelo global temporário para avaliação.
            model = create_model(random_state=seed)

            # Carrega parâmetros agregados do servidor.
            set_model_params(model, parameters, X_boot, y_boot, all_classes)

            # Avalia no conjunto global.
            acc, prec, rec, f1 = compute_metrics(model, X_global_eval, y_global_eval)

            # Salva no histórico.
            history["round"].append(server_round)
            history["accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1"].append(f1)

            # Log no terminal.
            log.info(
                "[EVAL][round=%s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
                server_round,
                acc,
                prec,
                rec,
                f1,
            )

            # Retorna loss aproximada e métricas.
            return float(1.0 - acc), {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }

        return evaluate

    # Garante que fraction_fit fique no intervalo [0,1].
    frac_fit = float(np.clip(fraction_fit, 0.0, 1.0))

    # Número total de clientes.
    num_clients = len(data.client_names)

    # Estratégia FedAvg do Flower.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=frac_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, int(np.ceil(frac_fit * num_clients))),
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        on_evaluate_config_fn=lambda server_round: {"server_round": server_round},
    )

    # Importa modulo ray
    import ray
    
    # Tenta usar Ray para simulação paralela.

    ray_init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "logging_level": "ERROR",
    }

    if ray_init_num_cpus is not None:
        ray_init_kwargs["num_cpus"] = ray_init_num_cpus

    if ray_init_num_gpus is not None:
        ray_init_kwargs["num_gpus"] = ray_init_num_gpus

    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)

    client_resources = {
        "num_cpus": client_num_cpus,
        "num_gpus": client_num_gpus,
    }

    # Importa a função de simulação do Flower.
    from flwr.simulation import start_simulation

    log.info(
        "[SIM] Iniciando simulação com %s clientes, %s rounds, fraction_fit=%.2f",
        num_clients,
        num_rounds,
        frac_fit,
    )

    # Inicia a simulação federada.
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Metadados da execução.
    setup = {
        "mode": "simulation",
        "clients": num_clients,
        "rounds": num_rounds,
        "fraction_fit": frac_fit,
        "local_epochs": local_epochs,
        "client_num_cpus": client_num_cpus,
        "client_num_gpus": client_num_gpus,
        "target_col": data.target_col,
        "feature_count": len(data.feature_names),
        "classes": data.classes,
        "global_eval_size": int(len(y_global_eval)),
    }

    # Salva resultados globais round a round.
    save_json_per_round(output_json, data.client_names, setup, history)

    # Consolida métricas dos clientes.
    aggregate_client_metrics(
        client_metrics_dir=os.path.join(run_dir, "client_metrics"),
        out_json=os.path.join(run_dir, "results_clients.json"),
        audit_csv=os.path.join(run_dir, "audit_clients.csv"),
        dataset_desc=data.client_names,
        setup=setup,
    )

    # Resumo final da execução.
    final = {
        "datasets": data.client_names,
        "final_accuracy": float(history["accuracy"][-1]) if history["accuracy"] else None,
        "final_precision": float(history["precision"][-1]) if history["precision"] else None,
        "final_recall": float(history["recall"][-1]) if history["recall"] else None,
        "final_f1": float(history["f1"][-1]) if history["f1"] else None,
    }

    return history, final


def parse_args():
    """
    Define e lê os argumentos de linha de comando.

    Por que usamos:
    - deixa o script reutilizável para diferentes execuções
    - permite trocar datasets, rounds, seed, eval_csv etc. sem editar código
    """

    p = argparse.ArgumentParser(description="FL com um dataset por cliente")
    p.add_argument("--client-csvs", type=str, required=True, help="Lista de CSVs dos clientes separada por vírgula")
    p.add_argument("--eval-csv", type=str, default=None, help="CSV opcional para avaliação global")
    p.add_argument("--target-col", type=str, default="class", help="Nome da coluna de rótulo")
    p.add_argument("--rounds", type=int, default=5, help="Número de rounds")
    p.add_argument("--fraction-fit", type=float, default=1.0, help="Fração de clientes por round")
    p.add_argument("--local-eval-size", type=float, default=0.2, help="Fração de cada dataset de cliente para avaliação local quando local_eval_source='split'")
    p.add_argument("--local-eval-source", type=str, choices=["split", "global"], default="split", help="Fonte da avaliação local dos clientes")
    p.add_argument("--seed", type=int, default=42, help="Semente RNG")
    p.add_argument("--run-id", type=str, default=None, help="Identificador da execução")
    p.add_argument("--local-epochs", type=int, default=1, help="Número de épocas locais por round")
    p.add_argument("--client-num-cpus", type=float, default=1.0, help="CPUs por cliente virtual")
    p.add_argument("--client-num-gpus", type=float, default=0.0, help="GPUs por cliente virtual")
    p.add_argument("--ray-init-num-cpus", type=int, default=None, help="Total de CPUs expostas ao Ray")
    p.add_argument("--ray-init-num-gpus", type=float, default=None, help="Total de GPUs expostas ao Ray")
    
    return p.parse_args()


def main():
    """
    Função principal do script.

    Fluxo geral:
    1. lê argumentos
    2. cria diretórios da execução
    3. prepara dados
    4. roda simulação FL
    5. gera plots globais
    6. salva resumo final
    """

    global run_dir

    # Lê argumentos CLI.
    args = parse_args()

    if args.local_epochs < 1:
        raise ValueError("--local-epochs deve ser >= 1")
    
    if args.client_num_cpus <= 0:
        raise ValueError("--client-num-cpus deve ser > 0")

    if args.client_num_gpus < 0:
        raise ValueError("--client-num-gpus deve ser >= 0")

    # Marca tempo inicial.
    start = time.time()

    # Define run_id automático se o usuário não informou.
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Diretório da execução.
    run_dir = os.path.join("runs", run_id)

    # Garante subpastas de saída.
    os.makedirs(os.path.join(run_dir, "docs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "client_metrics"), exist_ok=True)

    # Converte string de CSVs em lista real.
    client_csvs = [item.strip() for item in args.client_csvs.split(",") if item.strip()]

    if not client_csvs:
        raise ValueError("Nenhum CSV de cliente foi informado em --client-csvs.")

    # Logs iniciais.
    log.info("Run outputs serão salvos em: %s", run_dir)
    log.info("CSVs dos clientes: %s", client_csvs)
    if args.eval_csv:
        log.info("CSV de avaliação global: %s", args.eval_csv)

    # Prepara dados:
    # - leitura dos CSVs
    # - one-hot encoding
    # - normalização
    # - splits locais (ou teste global para avaliação local, dependendo da configuração)
    # - avaliação global
    data = prepare_federated_data(
        client_csvs=client_csvs,
        target_col=args.target_col,
        local_eval_size=args.local_eval_size,
        random_state=args.seed,
        eval_csv=args.eval_csv,
        local_eval_source=args.local_eval_source,
    )

    # Executa simulação federada.
    history, final = run_federated_simulation(
        data=data,
        output_json=os.path.join(run_dir, "results.json"),
        num_rounds=args.rounds,
        fraction_fit=args.fraction_fit,
        seed=args.seed,
        local_epochs=args.local_epochs,
        client_num_cpus=args.client_num_cpus,
        client_num_gpus=args.client_num_gpus,
        ray_init_num_cpus=args.ray_init_num_cpus,
        ray_init_num_gpus=args.ray_init_num_gpus,
    )

    # Gera plots globais automáticos.
    plot_metrics(history)

    # Salva um resumo consolidado da execução.
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "client_csvs": client_csvs,
                "eval_csv": args.eval_csv,
                "local_eval_size": args.local_eval_size,
                "feature_names": data.feature_names,
                "classes": data.classes,
                "final": final,
                "local_eval_source": args.local_eval_source, # Adiciona a fonte da avaliação local ao summary final
                "local_epochs": args.local_epochs,
                "client_num_cpus": args.client_num_cpus,
                "client_num_gpus": args.client_num_gpus,
                "ray_init_num_cpus": args.ray_init_num_cpus,
                "ray_init_num_gpus": args.ray_init_num_gpus,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Logs finais.
    log.info("=== Resultado final ===")
    log.info("Accuracy final: %s", final.get("final_accuracy"))
    log.info("Precision final: %s", final.get("final_precision"))
    log.info("Recall final: %s", final.get("final_recall"))
    log.info("F1 final: %s", final.get("final_f1"))
    log.info("Concluído em %.1fs", time.time() - start)


# Ponto de entrada do script.
# Garante que main() só rode quando o arquivo for executado diretamente.
if __name__ == "__main__":
    main()