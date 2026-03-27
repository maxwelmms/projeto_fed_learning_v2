#!/usr/bin/env python
# -*- coding: utf-8 -*-

# io é usado para criar um buffer de texto em memória.
# Aqui ele é útil para reconstruir o conteúdo do CSV após filtrarmos linhas indesejadas.
import io

# os é usado para verificar se arquivos existem e para extrair nomes de arquivos.
import os

# dataclass simplifica a criação de classes usadas como contêiner de dados.
from dataclasses import dataclass

# Tipagem opcional, para deixar o contrato das funções mais claro.
from typing import List, Optional, Sequence, Tuple

# NumPy é usado para arrays numéricos, conversões e concatenações.
import numpy as np

# pandas é usado para leitura e transformação tabular dos CSVs.
import pandas as pd

# train_test_split é usado para separar treino e teste local de cada cliente.
from sklearn.model_selection import train_test_split

# StandardScaler padroniza as features numéricas.
from sklearn.preprocessing import StandardScaler


@dataclass
class FederatedDataBundle:
    """
    Estrutura principal que carrega tudo o que o analyze.py precisa.

    O que ela guarda:
    - client_names: nome de cada dataset/cliente
    - client_train: lista de tuplas (X_train, y_train) por cliente
    - client_eval: lista de tuplas (X_test, y_test) por cliente
    - global_eval: conjunto global de avaliação
    - feature_names: nomes finais das features após one-hot
    - classes: lista de rótulos/classes detectadas globalmente
    - target_col: nome da coluna de rótulo
    - scaler: objeto StandardScaler já ajustado

    Por que usamos:
    - evita ficar retornando vários objetos separados
    - centraliza o pipeline de dados em uma estrutura só
    - facilita manutenção e entendimento do fluxo
    """

    client_names: List[str]
    client_train: List[Tuple[np.ndarray, np.ndarray]]
    client_eval: List[Tuple[np.ndarray, np.ndarray]]
    global_eval: Tuple[np.ndarray, np.ndarray]
    feature_names: List[str]
    classes: List[str]
    target_col: str
    scaler: StandardScaler


class LabelEncoderLite:
    """
    Encoder simples de rótulos, feito manualmente.

    O que faz:
    - recebe a lista global de classes
    - cria um mapeamento classe -> índice inteiro
    - converte labels textuais em inteiros

    Por que usamos:
    - precisamos que todos os clientes usem exatamente o mesmo mapeamento de classes
    - isso evita inconsistência entre clientes ao treinar e agregar modelos
    - é mais transparente do que usar um LabelEncoder externo aqui
    """

    def __init__(self, classes: Sequence[str]):
        # Guarda as classes como array.
        self.classes_ = np.array(list(classes), dtype=object)

        # Cria dicionário do tipo:
        # {"normal": 0, "attack": 1, ...}
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, values: Sequence[object]) -> np.ndarray:
        """
        Converte uma sequência de rótulos em índices inteiros.

        Exemplo:
        ["normal", "attack", "normal"] -> [0, 1, 0]

        Por que usamos:
        - modelos de ML normalmente trabalham com rótulos numéricos
        - o pipeline federado precisa de consistência entre clientes
        """
        out = []

        for value in values:
            # Se aparecer uma classe fora do universo conhecido, interrompe com erro.
            # Isso protege contra datasets inconsistentes.
            if value not in self.class_to_idx_:
                raise ValueError(
                    f"Rótulo desconhecido: {value!r}. "
                    f"Classes conhecidas: {list(self.classes_)}"
                )
            out.append(self.class_to_idx_[value])

        return np.asarray(out, dtype=np.int64)


def _read_csv_robusto(csv_path: str) -> pd.DataFrame:
    """
    Lê um CSV de forma mais tolerante.

    O que faz:
    - abre o arquivo em modo texto
    - ignora erros de encoding
    - remove linhas vazias
    - remove linhas começadas por '@' ou '%'
    - reconstrói o conteúdo em memória
    - passa esse conteúdo ao pandas

    Por que usamos:
    - alguns datasets podem conter metadados ou comentários no início
    - isso é comum em arquivos exportados de certos formatos
    - evita que o pd.read_csv quebre ou interprete mal o arquivo
    """

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    filtered = []

    for ln in lines:
        s = ln.strip()

        # Ignora linhas em branco.
        if not s:
            continue

        # Ignora linhas de metadados/comentários que começam com @ ou %.
        if s.startswith("@") or s.startswith("%"):
            continue

        filtered.append(ln)

    # Recria o conteúdo limpo em memória.
    buf = io.StringIO("".join(filtered))

    # Lê com pandas.
    # low_memory=False evita inferência fragmentada de tipos em arquivos maiores.
    return pd.read_csv(buf, low_memory=False)


def _load_single_frame(csv_path: str, target_col: str) -> pd.DataFrame:
    """
    Carrega e valida um único CSV.

    O que faz:
    - verifica se o arquivo existe
    - lê o CSV com o leitor robusto
    - valida se a coluna target existe
    - remove linhas sem rótulo

    Por que usamos:
    - cada cliente é representado por um CSV
    - precisamos garantir que todos tenham formato minimamente válido
    """

    # Verifica existência do arquivo antes de tentar ler.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    # Lê o CSV.
    df = _read_csv_robusto(csv_path)

    # Garante que a coluna de rótulo existe.
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna de rótulo '{target_col}' não está presente no arquivo {csv_path}.\n"
            f"Colunas detectadas: {list(df.columns)}"
        )

    # Remove linhas cujo rótulo está ausente.
    # Isso evita problemas de treino/avaliação depois.
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return df


def _build_feature_matrix(frames: Sequence[pd.DataFrame], target_col: str) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Constrói um espaço de features unificado para todos os datasets.

    O que faz:
    1. remove a coluna target de cada dataframe
    2. concatena todas as features em um dataframe só
    3. aplica tratamento de colunas categóricas
    4. aplica one-hot encoding global
    5. substitui inf/-inf por NaN
    6. preenche NaN com 0
    7. separa novamente cada bloco já codificado

    Por que usamos:
    - em FL com um dataset por cliente, os clientes podem ter categorias diferentes
    - se fizermos one-hot separadamente em cada cliente, os vetores terão colunas diferentes
    - o servidor não consegue agregar parâmetros se o espaço de features não for idêntico
    - então construímos um schema global de features antes
    """

    # Remove a coluna de rótulo de cada dataframe.
    features = [df.drop(columns=[target_col]).copy() for df in frames]

    # Junta tudo em um único dataframe para aprender o espaço global de colunas.
    combined = pd.concat(features, axis=0, ignore_index=True)

    # Para colunas do tipo objeto (strings/categóricas),
    # convertemos tudo para string e substituímos ausentes por marcador textual.
    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = combined[col].astype(str).fillna("<NA>")

    # One-hot encoding global.
    # drop_first=False preserva todas as categorias.
    # Isso é importante porque queremos um espaço completo e estável de atributos.
    combined = pd.get_dummies(combined, drop_first=False)

    # Substitui infinitos por NaN e NaN por zero.
    # Isso evita quebra no scaler e no modelo.
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Guarda os nomes finais das features.
    feature_names = combined.columns.tolist()

    # Agora precisamos separar novamente os blocos correspondentes a cada dataframe original.
    encoded_frames: List[pd.DataFrame] = []
    start = 0

    for feature_df in features:
        end = start + len(feature_df)
        encoded_frames.append(combined.iloc[start:end].reset_index(drop=True))
        start = end

    return encoded_frames, feature_names


def prepare_federated_data(
    client_csvs: Sequence[str],
    target_col: str = "class",
    local_eval_size: float = 0.2,
    random_state: int = 42,
    eval_csv: Optional[str] = None,
    local_eval_source: str = "split",
) -> FederatedDataBundle:
    """
    Função principal do pipeline de dados para o cenário federado.

    Responsabilidades:
    - carregar múltiplos CSVs, sendo um dataset exclusivo por cliente
    - opcionalmente carregar um CSV global de avaliação (eval_csv)
    - construir um espaço global de features consistente (one-hot alinhado)
    - codificar os rótulos de forma global e consistente entre clientes
    - preparar os dados locais de cada cliente:
        * modo "split": divide cada dataset local em treino e teste (holdout local)
        * modo "global": usa TODO o dataset do cliente para treino e define
          o mesmo dataset global (eval_csv) como teste local de todos os clientes
    - ajustar um scaler global com base nos dados de treino de todos os clientes
    - aplicar o scaler nos dados de treino, avaliação local e avaliação global
    - construir um conjunto de avaliação global:
        * usando eval_csv (se fornecido), ou
        * concatenando os testes locais (modo split)

    Parâmetros importantes:
    - client_csvs: lista de caminhos (1 CSV por cliente)
    - eval_csv: dataset opcional para avaliação global padronizada
    - local_eval_source:
        * "split"  → avaliação local baseada no próprio dataset do cliente
        * "global" → avaliação local baseada no mesmo dataset global (eval_csv)

    Retorno:
    - FederatedDataBundle contendo:
        * dados de treino por cliente
        * dados de avaliação por cliente
        * dataset de avaliação global
        * metadados (features, classes, scaler)

    Observação:
    - Os dados dos clientes NÃO são misturados para treino.
    - O alinhamento global de features e o scaler garantem compatibilidade
      entre os modelos locais no contexto federado.

    Essa função é o núcleo do pipeline de dados do sistema.
    """

    # Garante que ao menos um cliente foi informado.
    if not client_csvs:
        raise ValueError("É necessário informar ao menos um dataset de cliente.")
    
    if local_eval_source not in {"split", "global"}:
        raise ValueError("local_eval_source deve ser 'split' ou 'global'.")

    # Carrega cada CSV de cliente como dataframe.
    client_frames = [_load_single_frame(path, target_col) for path in client_csvs]

    # Se houver CSV de avaliação global, carrega também.
    eval_frame = _load_single_frame(eval_csv, target_col) if eval_csv else None

    # Garante que se local_eval_source for "global", o eval_csv foi fornecido.
    if local_eval_source == "global" and eval_frame is None:
        raise ValueError(
        "Quando local_eval_source='global', é obrigatório informar --eval-csv."
    )

    # Conjunto de dataframes que será usado para aprender o schema global de features.
    # Incluímos o eval_frame aqui para garantir alinhamento completo de colunas,
    # caso o dataset de avaliação tenha categorias que não apareceram no treino.
    frames_for_schema = client_frames + ([eval_frame] if eval_frame is not None else [])

    # Faz one-hot global e retorna versões codificadas de cada dataframe.
    encoded_frames, feature_names = _build_feature_matrix(frames_for_schema, target_col)

    # Os primeiros blocos correspondem aos clientes.
    encoded_clients = encoded_frames[: len(client_frames)]

    # O último bloco, se existir, corresponde ao eval_csv.
    encoded_eval = encoded_frames[-1] if eval_frame is not None else None

    # Descobre todas as classes globais olhando todos os dataframes.
    # Isso garante consistência entre clientes e avaliação.
    all_labels = pd.concat([df[target_col] for df in frames_for_schema], axis=0).astype(str)

    # Ordena as classes para ter mapeamento estável.
    classes = sorted(all_labels.unique().tolist())

    # Cria encoder global de rótulos.
    label_encoder = LabelEncoderLite(classes)

    # Prepara a versão bruta/codificada do dataset global de avaliação, se eval_csv foi fornecido.
    global_X_raw = None
    global_y_raw = None

    if eval_frame is not None and encoded_eval is not None:
        global_X_raw = encoded_eval.to_numpy(dtype=np.float64, copy=True)
        global_y_raw = label_encoder.transform(eval_frame[target_col].astype(str).tolist())

    # Estruturas que irão armazenar treino e teste por cliente.
    client_train: List[Tuple[np.ndarray, np.ndarray]] = []
    client_eval: List[Tuple[np.ndarray, np.ndarray]] = []

    # Estruturas auxiliares para ajustar o scaler global no treino.
    train_blocks: List[np.ndarray] = []

    # Estruturas auxiliares para montar a avaliação global quando não houver eval_csv externo.
    eval_blocks: List[np.ndarray] = []
    eval_labels: List[np.ndarray] = []

    # Percorre cada cliente e seu dataframe codificado.
    for idx, (raw_df, enc_df) in enumerate(zip(client_frames, encoded_clients)):
        # Converte features para matriz NumPy float64.
        X = enc_df.to_numpy(dtype=np.float64, copy=True)

        # Converte rótulos textuais em inteiros globais.
        y = label_encoder.transform(raw_df[target_col].astype(str).tolist())

        # Verifica qual é a fonte da avaliação local do cliente.
        # Se for "split", o teste local será criado a partir do próprio dataset do cliente.
        if local_eval_source == "split":

            # Define se é possível fazer split treino/teste.
            # Isso só é viável se:
            # - o usuário pediu uma fração de teste maior que zero
            # - o dataset do cliente tem pelo menos 2 amostras
            can_split = local_eval_size > 0.0 and len(X) >= 2

            # Se for possível dividir o dataset em treino e teste:
            if can_split:
                try:
                    # Define se o split será estratificado.
                    # Se houver mais de uma classe em y, usamos stratify=y
                    # para preservar aproximadamente a distribuição de classes.
                    # Se houver só uma classe, stratify não pode ser usado.
                    stratify = y if len(np.unique(y)) > 1 else None

                    # Faz a divisão treino/teste do dataset local do cliente.
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X,
                        y,
                        test_size=local_eval_size,
                        random_state=random_state + idx,
                        stratify=stratify,
                    )

                except Exception:
                    # Fallback manual se o split estratificado falhar.
                    cut = max(1, int(round((1.0 - local_eval_size) * len(X))))
                    cut = min(cut, len(X) - 1)

                    X_tr, X_te = X[:cut], X[cut:]
                    y_tr, y_te = y[:cut], y[cut:]

            else:
                # Se não for possível fazer split, todo o dataset vira treino.
                X_tr, y_tr = X, y
                X_te = np.empty((0, X.shape[1]), dtype=np.float64)
                y_te = np.empty((0,), dtype=np.int64)

            # Salva treino e avaliação local do cliente.
            client_train.append((X_tr, y_tr))
            client_eval.append((X_te, y_te))
            train_blocks.append(X_tr)

            # Se existir teste local, acumula para possível avaliação global.
            if len(X_te) > 0:
                eval_blocks.append(X_te)
                eval_labels.append(y_te)

        else:
            # local_eval_source == "global"
            # Todo o dataset do cliente vira treino local.
            X_tr, y_tr = X, y
            client_train.append((X_tr, y_tr))
            train_blocks.append(X_tr)

            # Todos os clientes usam o mesmo dataset global para avaliação local.
            client_eval.append((global_X_raw.copy(), global_y_raw.copy()))

    # Cria scaler global.
    scaler = StandardScaler()

    # Junta todos os blocos de treino local.
    X_train_all = np.vstack(train_blocks)

    # Ajusta o scaler APENAS nos dados de treino.
    # Isso é importante para evitar vazamento de informação do teste.
    scaler.fit(X_train_all)

    # Escala os treinos locais.
    client_train_scaled = [(scaler.transform(Xtr), ytr) for Xtr, ytr in client_train]

    # Escala as avaliações locais, quando existirem.
    client_eval_scaled = [
        (scaler.transform(Xte), yte) if len(Xte) > 0 else (Xte, yte)
        for Xte, yte in client_eval
    ]

    # Caso exista eval_csv externo:
    if eval_frame is not None and encoded_eval is not None:
        # Escala usando o mesmo scaler global treinado nos treinos locais.
        X_eval = scaler.transform(encoded_eval.to_numpy(dtype=np.float64, copy=True))

        # Codifica rótulos com o mesmo encoder global.
        y_eval = label_encoder.transform(eval_frame[target_col].astype(str).tolist())

        global_eval = (X_eval, y_eval)

    else:
        # Se não existe eval_csv, usamos a concatenação dos holdouts locais.
        if not eval_blocks:
            raise ValueError(
                "Nenhum conjunto de avaliação global disponível. "
                "Use --local-eval-size > 0 ou informe --eval-csv."
            )

        X_eval = scaler.transform(np.vstack(eval_blocks))
        y_eval = np.concatenate(eval_labels)
        global_eval = (X_eval, y_eval)

    # Extrai apenas o nome do arquivo de cada cliente.
    # Isso é útil para logs e metadados da execução.
    client_names = [os.path.basename(path) for path in client_csvs]

    # Retorna o pacote completo.
    return FederatedDataBundle(
        client_names=client_names,
        client_train=client_train_scaled,
        client_eval=client_eval_scaled,
        global_eval=global_eval,
        feature_names=feature_names,
        classes=classes,
        target_col=target_col,
        scaler=scaler,
    )