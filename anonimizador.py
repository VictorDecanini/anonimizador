"""
Anonimizador de grandes volumes de dados (CSV/Excel) — Scanntech.

Anonimiza fornecedor, ean, marca, sku, canal e uf. As demais colunas ficam
intactas. Os códigos gerados são sempre numéricos. Processa em blocos para
suportar arquivos de 600MB+ sem carregar tudo na memória de uma vez.

Fluxo de anonimização:
  1. Sobe o arquivo.
  2. Analisa (detecta colunas-alvo e escaneia valores únicos das colunas
     categóricas filtráveis: fornecedor, marca, canal, uf).
  3. Escolhe quais colunas anonimizar e, opcionalmente, filtra valores
     específicos (ex: só um fabricante) para direcionar a análise.
  4. Anonimiza só a base filtrada.
"""

import os
import re
import time
import tempfile

import pandas as pd
import streamlit as st
import unicodedata
from charset_normalizer import from_bytes

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CHUNK_SIZE = 200_000  # linhas processadas por bloco

COLUNAS_ALVO = {
    1: {
        "nome": "fornecedor",
        "aliases": [
            "fornecedor", "fabricante", "proveedor", "supplier", "manufacturer",
            "nome fornecedor", "razao social fornecedor", "razao social",
            "nome fabricante", "cod fornecedor", "codigo fornecedor",
        ],
    },
    2: {
        "nome": "ean",
        "aliases": [
            "ean", "codigo ean", "cod ean", "codigo de barras", "codigo barras",
            "barcode", "gtin", "nome ean",
        ],
    },
    3: {
        "nome": "marca",
        "aliases": ["marca", "brand", "nome marca"],
    },
    4: {
        "nome": "sku",
        "aliases": [
            "sku", "nome sku", "cod sku", "codigo sku", "descricao sku",
            "produto", "nome produto", "descricao produto", "descricao",
            "nome do produto", "item", "nome item", "nome do item",
        ],
    },
    5: {
        "nome": "canal",
        "aliases": ["canal", "channel", "pdv canal", "tipo canal"],
    },
    6: {
        "nome": "uf",
        "aliases": ["uf", "estado", "state", "unidade federativa"],
    },
}
BASE_INDICE = 10**9  # até 999.999.999 valores únicos por coluna

# Colunas oferecidas no filtro de valores (categóricas, baixa cardinalidade).
# EAN e SKU ficam de fora do filtro por terem alta cardinalidade -- um
# multiselect com milhões de opções não seria usável.
COLUNAS_FILTRAVEIS = {"fornecedor", "marca", "canal", "uf", "sku"}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def normalizar(texto):
    texto = str(texto).strip().lower()
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def normalizar_chave(texto):
    """Remove acento, minusculiza e tira espaço/underscore/hífen/pontuação
    -- assim 'Nome SKU', 'nome_sku' e 'Nome-Sku' todos virem 'nomesku' e
    casam entre si, independente de como a coluna foi escrita."""
    texto = normalizar(texto)
    return re.sub(r"[^a-z0-9]", "", texto)


def padronizar_canal(valor):
    v = normalizar(valor).replace(" ", "_").replace("-", "_")
    if v in ("1_a_4", "1_4", "1a4"):
        return "canal_1_4"
    if v in ("5_a_9", "5_9", "5a9"):
        return "canal_5_9"
    if v in ("10", "10_mais", "10_ou_mais", "10plus", "10+"):
        return "canal_10_mais"
    if v in ("atacarejo", "atacado", "cash_carry", "cash_and_carry"):
        return "canal_atacarejo"
    return v


def identificar_colunas(colunas_arquivo):
    """Casa colunas do arquivo com colunas-alvo. Primeiro tenta um match
    exato (ignorando espaço/acento/maiúscula); se não achar, cai para um
    fallback por substring (só com aliases de 4+ caracteres, pra não dar
    falso positivo com termos curtos como 'uf')."""
    aliases_por_indice = {
        indice: [normalizar_chave(a) for a in info["aliases"]]
        for indice, info in COLUNAS_ALVO.items()
    }
    todos_aliases = sorted(
        (
            (alias, indice)
            for indice, aliases in aliases_por_indice.items()
            for alias in aliases
        ),
        key=lambda x: -len(x[0]),
    )

    encontradas = {}
    for coluna in colunas_arquivo:
        col_norm = normalizar_chave(coluna)
        indice_encontrado = None

        for indice, aliases in aliases_por_indice.items():
            if col_norm in aliases:
                indice_encontrado = indice
                break

        if indice_encontrado is None:
            for alias, indice in todos_aliases:
                if len(alias) >= 4 and alias in col_norm:
                    indice_encontrado = indice
                    break

        if indice_encontrado is not None:
            encontradas[coluna] = indice_encontrado

    return encontradas


def carregar_mapa(arquivo_mapa):
    """Carrega um mapeamento existente (coluna, valor_original, codigo), se enviado."""
    mapas = {i: {} for i in COLUNAS_ALVO}
    proximo_seq = {i: 1 for i in COLUNAS_ALVO}
    nome_para_indice = {info["nome"]: i for i, info in COLUNAS_ALVO.items()}

    if arquivo_mapa is not None:
        df_mapa = pd.read_csv(arquivo_mapa, dtype=str)
        for _, linha in df_mapa.iterrows():
            indice = nome_para_indice.get(linha["coluna"])
            if indice is None:
                continue
            chave = linha["valor_original"]
            codigo = int(linha["codigo"])
            mapas[indice][chave] = codigo
            seq = codigo - indice * BASE_INDICE
            proximo_seq[indice] = max(proximo_seq[indice], seq + 1)

    return mapas, proximo_seq


def anonimizar_coluna(serie, indice, mapas, proximo_seq):
    """Substitui valores por códigos numéricos, normalizando apenas os
    valores ÚNICOS do bloco (não célula a célula) para performance."""
    nome_coluna = COLUNAS_ALVO[indice]["nome"]
    eh_canal = nome_coluna == "canal"
    mapa_coluna = mapas[indice]

    chave_por_valor = {}
    for v in serie.unique():
        v_str = "" if pd.isna(v) else str(v).strip()
        if v_str == "":
            continue
        chave_por_valor[v] = padronizar_canal(v_str) if eh_canal else normalizar(v_str)

    for chave in set(chave_por_valor.values()):
        if chave not in mapa_coluna:
            codigo = indice * BASE_INDICE + proximo_seq[indice]
            mapa_coluna[chave] = codigo
            proximo_seq[indice] += 1

    codigo_por_valor = {v: mapa_coluna[chave] for v, chave in chave_por_valor.items()}
    resultado = serie.map(codigo_por_valor)

    # .map() força a coluna inteira para float64 quando há NaN misturado
    # com os códigos inteiros -- por isso reconstruímos como object.
    resultado = pd.Series(
        [None if pd.isna(x) else int(x) for x in resultado],
        index=resultado.index, dtype="object"
    )
    resultado = resultado.where(resultado.notna(), serie)  # mantém em branco/NaN como estava

    return resultado


def tentar_numerico(serie):
    """Converte para número real qualquer valor restaurado que 'parece
    número' (ex: EAN), preservando texto genuíno (ex: nome de marca)."""
    convertido = pd.to_numeric(serie, errors="coerce")
    valores = []
    for original, conv in zip(serie, convertido):
        if pd.isna(conv):
            valores.append(original)
        elif float(conv).is_integer():
            valores.append(int(conv))
        else:
            valores.append(conv)
    return pd.Series(valores, index=serie.index, dtype="object")


def desanonimizar_coluna(serie, mapa_coluna):
    """Troca os códigos numéricos pelos valores originais, usando o
    mapeamento (coluna -> {codigo: valor_original})."""
    restaurado = serie.map(mapa_coluna)
    restaurado = restaurado.where(restaurado.notna(), serie)
    return tentar_numerico(restaurado)


def construir_df_mapa(mapas):
    # MELHORIA: valor_original sempre em MAIÚSCULA na referência, para ficar
    # mais legível quando for desanonimizar depois. A chave interna do
    # dicionário (usada para deduplicar) continua normalizada em minúscula
    # -- isso aqui só afeta o texto exibido/exportado.
    linhas = [
        {"coluna": COLUNAS_ALVO[indice]["nome"], "valor_original": chave.upper(), "codigo": codigo}
        for indice, mapa in mapas.items()
        for chave, codigo in mapa.items()
    ]
    return pd.DataFrame(linhas)


def formatar_tempo(segundos):
    segundos = max(int(segundos), 0)
    if segundos < 60:
        return f"{segundos}s"
    m, s = divmod(segundos, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def detectar_encoding(arquivo, tamanho_amostra=200_000):
    """Detecta automaticamente a codificação do arquivo a partir de uma
    amostra dos bytes -- evita ficar testando utf-8/latin-1/cp1252 na mão."""
    arquivo.seek(0)
    amostra = arquivo.read(tamanho_amostra)
    arquivo.seek(0)
    resultado = from_bytes(amostra).best()
    return resultado.encoding if resultado else "utf-8-sig"


def eh_excel(arquivo):
    return arquivo.name.lower().endswith((".xlsx", ".xls"))


def ler_colunas_amostra(arquivo, sep, encoding):
    arquivo.seek(0)
    if eh_excel(arquivo):
        amostra = pd.read_excel(arquivo, nrows=1000, dtype=str)
    else:
        amostra = pd.read_csv(arquivo, sep=sep, encoding=encoding, nrows=1000, dtype=str)
    arquivo.seek(0)
    return amostra.columns


def ler_em_blocos(arquivo, sep, encoding):
    """Gera blocos do arquivo. CSV/TXT: em blocos reais (leve na memória,
    sustenta 600MB+). Excel: lido de uma vez só, já que o pandas não tem
    leitura em blocos nativa para xlsx -- para arquivos MUITO grandes,
    prefira CSV."""
    arquivo.seek(0)
    if eh_excel(arquivo):
        yield pd.read_excel(arquivo, dtype=str)
    else:
        leitor = pd.read_csv(
            arquivo, sep=sep, encoding=encoding, chunksize=CHUNK_SIZE,
            dtype=str, keep_default_na=False
        )
        for chunk in leitor:
            yield chunk


def analisar_arquivo(arquivo, sep, encoding):
    """Varre o arquivo (em blocos) para: (1) listar todas as colunas do
    arquivo, (2) identificar quais são colunas-alvo, e (3) coletar valores
    únicos das colunas categóricas filtráveis (fornecedor, marca, canal,
    uf, sku)."""
    colunas_arquivo = list(ler_colunas_amostra(arquivo, sep, encoding))
    colunas_alvo = identificar_colunas(colunas_arquivo)

    colunas_filtro = {
        col: COLUNAS_ALVO[idx]["nome"]
        for col, idx in colunas_alvo.items()
        if COLUNAS_ALVO[idx]["nome"] in COLUNAS_FILTRAVEIS
    }

    valores_unicos = {col: set() for col in colunas_filtro}

    if colunas_filtro:
        for chunk in ler_em_blocos(arquivo, sep, encoding):
            for col in colunas_filtro:
                if col in chunk.columns:
                    valores = chunk[col].dropna().unique()
                    valores_unicos[col].update(v for v in valores if str(v).strip() != "")

    valores_unicos_ordenados = {c: sorted(v, key=str) for c, v in valores_unicos.items()}
    return colunas_arquivo, colunas_alvo, colunas_filtro, valores_unicos_ordenados


# ---------------------------------------------------------------------------
# Processamento: anonimização
# ---------------------------------------------------------------------------
def processar_anonimizacao(arquivo, sep, encoding, colunas_finais=None, filtros=None):
    tamanho_total = arquivo.size
    mapas, proximo_seq = carregar_mapa(None)

    colunas_arquivo = list(ler_colunas_amostra(arquivo, sep, encoding))
    colunas_alvo = identificar_colunas(colunas_arquivo)

    # As colunas categóricas detectadas (fornecedor, ean, marca, sku, canal,
    # uf) são SEMPRE anonimizadas -- não é uma escolha do usuário. O que o
    # usuário escolhe é quais colunas (de todas do arquivo) vão para o
    # resultado final; se uma coluna-alvo for excluída do resultado, ela
    # simplesmente não aparece (não há o que anonimizar nela).
    if colunas_finais:
        colunas_saida = [c for c in colunas_arquivo if c in colunas_finais]
        colunas_alvo = {col: idx for col, idx in colunas_alvo.items() if col in colunas_finais}
    else:
        colunas_saida = colunas_arquivo

    if not colunas_alvo:
        st.warning("Nenhuma coluna-alvo (fornecedor, ean, marca, sku, canal, uf) está incluída no arquivo final.")
        return

    st.info(
        "Colunas anonimizadas: "
        + ", ".join(f"**{col}** → {COLUNAS_ALVO[i]['nome']}" for col, i in colunas_alvo.items())
    )

    if filtros:
        descricao_filtros = "; ".join(f"{nome_col} em {valores}" for nome_col, valores in filtros.items())
        st.info(f"🔎 Filtro aplicado: {descricao_filtros}")

    status = st.empty()
    barra = st.progress(0.0)
    inicio = time.time()

    saida_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
    linhas_lidas = 0
    linhas_mantidas = 0
    primeiro_chunk = True

    for chunk in ler_em_blocos(arquivo, sep, encoding):
        linhas_lidas += len(chunk)

        if filtros:
            mascara = pd.Series(True, index=chunk.index)
            for col, valores in filtros.items():
                if col in chunk.columns:
                    valores_str = {str(v) for v in valores}
                    mascara &= chunk[col].astype(str).isin(valores_str)
            chunk = chunk[mascara]

        if not chunk.empty:
            for coluna, indice in colunas_alvo.items():
                if coluna in chunk.columns:
                    chunk[coluna] = anonimizar_coluna(chunk[coluna], indice, mapas, proximo_seq)

            colunas_presentes = [c for c in colunas_saida if c in chunk.columns]
            chunk = chunk[colunas_presentes]

            chunk.to_csv(saida_tmp, sep=sep, index=False, header=primeiro_chunk)
            primeiro_chunk = False
            linhas_mantidas += len(chunk)

        fracao = min(arquivo.tell() / tamanho_total, 1.0) if tamanho_total else 0.0
        barra.progress(fracao)
        elapsed = time.time() - inicio
        eta_txt = formatar_tempo(elapsed / fracao - elapsed) if fracao > 0.02 else "calculando..."
        status.text(
            f"Anonimizando... {fracao * 100:.1f}% — {linhas_mantidas:,} linhas mantidas "
            f"de {linhas_lidas:,} lidas — tempo restante: {eta_txt}"
        )

    saida_tmp.close()
    barra.progress(1.0)

    if primeiro_chunk:
        status.text("Concluído: nenhuma linha correspondeu ao filtro.")
        st.warning("Nenhuma linha do arquivo correspondeu ao filtro selecionado. Ajuste os filtros e tente novamente.")
        return

    status.text(
        f"Concluído: {linhas_mantidas:,} linhas no arquivo final "
        f"(de {linhas_lidas:,} lidas) em {formatar_tempo(time.time() - inicio)}."
    )

    df_mapa = construir_df_mapa(mapas)

    st.session_state["resultado_anon"] = {
        "csv_path": saida_tmp.name,
        "mapa_csv_bytes": df_mapa.to_csv(index=False).encode("utf-8-sig"),
        "total_linhas": linhas_mantidas,
        "total_lidas": linhas_lidas,
        "total_mapeados": len(df_mapa),
    }


# ---------------------------------------------------------------------------
# Processamento: desanonimização
# ---------------------------------------------------------------------------
def processar_desanonimizacao(arquivo, sep, encoding_manual, arquivo_mapa):
    if arquivo_mapa is None:
        st.warning("Envie o arquivo de mapeamento (referência) para desanonimizar.")
        return

    tamanho_total = arquivo.size
    df_mapa = pd.read_csv(arquivo_mapa, dtype=str)
    mapa_reverso = {
        coluna: dict(zip(grupo["codigo"].astype(str), grupo["valor_original"]))
        for coluna, grupo in df_mapa.groupby("coluna")
    }

    if eh_excel(arquivo):
        encoding = None
    elif encoding_manual == "Automático":
        encoding = detectar_encoding(arquivo)
        st.caption(f"🔎 Codificação detectada automaticamente: **{encoding}**")
    else:
        encoding = encoding_manual

    colunas_arquivo = ler_colunas_amostra(arquivo, sep, encoding)
    colunas_no_arquivo = identificar_colunas(colunas_arquivo)
    colunas_para_restaurar = {
        col: COLUNAS_ALVO[idx]["nome"]
        for col, idx in colunas_no_arquivo.items()
        if COLUNAS_ALVO[idx]["nome"] in mapa_reverso
    }

    if not colunas_para_restaurar:
        st.warning("Nenhuma coluna do arquivo bate com o mapeamento enviado.")
        return

    st.info("Colunas a restaurar: " + ", ".join(f"**{c}**" for c in colunas_para_restaurar))

    status = st.empty()
    barra = st.progress(0.0)
    inicio = time.time()

    saida_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
    linhas_processadas = 0
    primeiro_chunk = True

    for chunk in ler_em_blocos(arquivo, sep, encoding):
        for coluna, nome_padrao in colunas_para_restaurar.items():
            chunk[coluna] = desanonimizar_coluna(chunk[coluna], mapa_reverso[nome_padrao])

        chunk.to_csv(saida_tmp, sep=sep, index=False, header=primeiro_chunk)
        primeiro_chunk = False
        linhas_processadas += len(chunk)

        fracao = min(arquivo.tell() / tamanho_total, 1.0) if tamanho_total else 0.0
        barra.progress(fracao)
        elapsed = time.time() - inicio
        eta_txt = formatar_tempo(elapsed / fracao - elapsed) if fracao > 0.02 else "calculando..."
        status.text(f"Desanonimizando... {fracao * 100:.1f}% — {linhas_processadas:,} linhas — tempo restante: {eta_txt}")

    saida_tmp.close()
    barra.progress(1.0)
    status.text(f"Concluído: {linhas_processadas:,} linhas em {formatar_tempo(time.time() - inicio)}.")

    st.session_state["resultado_desanon"] = {
        "csv_path": saida_tmp.name,
        "total_linhas": linhas_processadas,
    }


# ---------------------------------------------------------------------------
# Interface (perfil executivo, clean)
# ---------------------------------------------------------------------------
CSS_EXECUTIVO = """
<style>
    .stApp {
        background-color: #F4F6F9;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B1F3A;
        padding-top: 0;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #F4F6F9 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: white;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0B1F3A !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #0B1F3A;
        color: white;
        border: none;
        padding: 10px 26px;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(11, 31, 58, 0.25);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #16406E;
        box-shadow: 0 4px 14px rgba(11, 31, 58, 0.35);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1.5px solid #0B1F3A;
        color: #0B1F3A;
        font-weight: 600;
        background-color: white;
    }
    .stDownloadButton > button:hover {
        background-color: #0B1F3A;
        color: white;
    }
    div[data-testid="stProgress"] > div > div {
        background-image: linear-gradient(90deg, #0B1F3A, #2F8FDB);
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        border-bottom: none !important;
    }
    .stTabs button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 10px 26px !important;
        background-color: #E8ECF3 !important;
        border: none !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    .stTabs button[data-baseweb="tab"] p {
        color: #0B1F3A !important;
        font-weight: 600 !important;
    }
    .stTabs button[aria-selected="true"] {
        background-color: #0B1F3A !important;
        box-shadow: 0 3px 10px rgba(11, 31, 58, 0.35) !important;
    }
    .stTabs button[aria-selected="true"] p {
        color: white !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
        box-shadow: none;
    }
    .block-container {
        padding-top: 1rem;
    }
    div[data-testid="stImage"] {
        position: fixed;
        top: 1.1rem;
        left: 23rem;
        z-index: 1000;
    }
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] p {
        text-align: center;
    }
</style>
"""

st.set_page_config(page_title="Anonimizador Scanntech", page_icon="🔒", layout="centered")
st.markdown(CSS_EXECUTIVO, unsafe_allow_html=True)

if os.path.exists("logo_scanntech.png"):
    st.image("logo_scanntech.png", width=260)

st.markdown(
    "<h1 style='color:#054FE1; margin-bottom:0;'>Anonimizador de Dados</h1>"
    "<p style='color:#5A6B85; margin-top:2px;'>Fornecedor · EAN · Marca · SKU · Canal · UF"
    " — códigos sempre numéricos</p>",
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    separador = st.selectbox("Separador do CSV (ignorado para Excel)", [";", ",", "\t", "|"], index=0)
    encoding = st.selectbox(
        "Codificação do CSV (ignorado para Excel)",
        ["Automático", "utf-8-sig", "utf-8", "cp1252", "latin1"],
        index=0,
    )

aba_anonimizar, aba_desanonimizar = st.tabs(["🔒 Anonimizar", "🔓 Desanonimizar"])

with aba_anonimizar:
    with st.container(border=True):
        arquivo = st.file_uploader(
            "Arquivo para anonimizar (CSV, TXT ou Excel)",
            type=["csv", "txt", "xlsx", "xls"], key="upload_anon",
        )

        if arquivo is not None:
            st.caption(f"📄 {arquivo.name} — {arquivo.size / 1_048_576:.1f} MB")

            if st.button("🔍 Analisar colunas e valores", key="btn_analisar_anon"):
                with st.spinner("Lendo colunas e valores únicos das colunas categóricas..."):
                    if eh_excel(arquivo):
                        encoding_usar = None
                    elif encoding == "Automático":
                        encoding_usar = detectar_encoding(arquivo)
                    else:
                        encoding_usar = encoding

                    colunas_arquivo, colunas_alvo, colunas_filtro, valores_unicos = analisar_arquivo(
                        arquivo, separador, encoding_usar
                    )

                st.session_state["analise_anon"] = {
                    "arquivo_nome": arquivo.name,
                    "colunas_arquivo": colunas_arquivo,
                    "colunas_alvo": colunas_alvo,
                    "colunas_filtro": colunas_filtro,
                    "valores_unicos": valores_unicos,
                    "encoding_usado": encoding_usar,
                }

    analise = st.session_state.get("analise_anon")
    if analise is not None and arquivo is not None and analise["arquivo_nome"] == arquivo.name:
        if not analise["colunas_alvo"]:
            st.warning("Nenhuma coluna-alvo (fornecedor, ean, marca, sku, canal, uf) foi encontrada neste arquivo.")
        else:
            nomes_categoricos_detectados = sorted({COLUNAS_ALVO[i]["nome"] for i in analise["colunas_alvo"].values()})

            with st.container(border=True):
                st.markdown("**1. Colunas que vão para o arquivo final**")
                st.caption(
                    "As colunas categóricas (" + ", ".join(nomes_categoricos_detectados)
                    + ") são sempre anonimizadas quando incluídas. Desmarque aqui só o que você não quer no resultado."
                )

                def _marcar_todas_colunas():
                    valor = st.session_state["marcar_todas_colunas_anon"]
                    for col in analise["colunas_arquivo"]:
                        st.session_state[f"col_final_{col}"] = valor

                st.checkbox(
                    "Selecionar todas", value=True, key="marcar_todas_colunas_anon",
                    on_change=_marcar_todas_colunas,
                )

                colunas_marcadas = {}
                cols_ui = st.columns(4)
                for i, col in enumerate(analise["colunas_arquivo"]):
                    with cols_ui[i % 4]:
                        chave_widget = f"col_final_{col}"
                        if chave_widget not in st.session_state:
                            st.session_state[chave_widget] = True
                        rotulo = col
                        if col in analise["colunas_alvo"]:
                            rotulo += " 🔒"  # marca visualmente as que serão anonimizadas
                        colunas_marcadas[col] = st.checkbox(rotulo, key=chave_widget)

                filtros_selecionados = {}
                if analise["colunas_filtro"]:
                    st.divider()
                    st.markdown("**2. Filtrar valores (opcional — vazio mantém tudo)**")
                    st.caption("Direcione a análise a um recorte específico, ex: só um fabricante ou uma marca.")

                    colunas_filtro_itens = list(analise["colunas_filtro"].items())
                    cols_filtro_ui = st.columns(min(len(colunas_filtro_itens), 4))
                    for i, (col, nome) in enumerate(colunas_filtro_itens):
                        with cols_filtro_ui[i % 4]:
                            opcoes = analise["valores_unicos"].get(col, [])
                            selecionados = st.multiselect(
                                nome.capitalize(), opcoes, key=f"filtro_anon_{col}"
                            )
                            if selecionados:
                                filtros_selecionados[col] = selecionados

                st.divider()
                st.markdown("**3. Anonimizar**")
                if st.button("▶️ Iniciar anonimização", type="primary", key="btn_anon"):
                    colunas_finais_selecionadas = {c for c, v in colunas_marcadas.items() if v}
                    if not colunas_finais_selecionadas:
                        st.warning("Selecione ao menos uma coluna para o arquivo final.")
                    else:
                        processar_anonimizacao(
                            arquivo, separador, analise["encoding_usado"],
                            colunas_finais_selecionadas, filtros_selecionados,
                        )

    if "resultado_anon" in st.session_state:
        resultado = st.session_state["resultado_anon"]
        st.write("")
        with st.container(border=True):
            st.success("✅ Anonimização concluída")
            c1, c2 = st.columns(2)
            c1.metric("Linhas no arquivo final", f"{resultado['total_linhas']:,}")
            c2.metric("Valores únicos mapeados", f"{resultado['total_mapeados']:,}")

            col_a, col_b = st.columns(2)
            with col_a:
                with open(resultado["csv_path"], "rb") as f:
                    st.download_button(
                        "📥 Baixar CSV anonimizado", f, file_name="dados_anonimizados.csv",
                        mime="text/csv", key="download_csv", use_container_width=True,
                    )
            with col_b:
                st.download_button(
                    "📥 Baixar mapeamento (referência)",
                    resultado["mapa_csv_bytes"], file_name="mapeamento_anonimizacao.csv",
                    mime="text/csv", key="download_mapa", use_container_width=True,
                )
            st.caption("Guarde o arquivo de referência — é ele que permite desanonimizar depois.")

with aba_desanonimizar:
    with st.container(border=True):
        arquivo_mapa = st.file_uploader("Mapeamento (referência)", type=["csv"], key="upload_mapa_desanon")
        arquivo_desanon = st.file_uploader(
            "Arquivo anonimizado (CSV, TXT ou Excel)",
            type=["csv", "txt", "xlsx", "xls"], key="upload_desanon",
        )

        if arquivo_desanon is not None:
            st.caption(f"📄 {arquivo_desanon.name} — {arquivo_desanon.size / 1_048_576:.1f} MB")
            if st.button("▶️ Iniciar desanonimização", type="primary", key="btn_desanon"):
                processar_desanonimizacao(arquivo_desanon, separador, encoding, arquivo_mapa)

    if "resultado_desanon" in st.session_state:
        resultado = st.session_state["resultado_desanon"]
        st.write("")
        with st.container(border=True):
            st.success("✅ Desanonimização concluída")
            st.metric("Linhas processadas", f"{resultado['total_linhas']:,}")
            with open(resultado["csv_path"], "rb") as f:
                st.download_button(
                    "📥 Baixar CSV desanonimizado", f, file_name="dados_desanonimizados.csv",
                    mime="text/csv", key="download_csv_desanon", use_container_width=True,
                )