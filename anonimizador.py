"""
Anonimizador de grandes volumes de dados (CSV) — Scanntech.

Anonimiza/desanonimiza fornecedor, ean, marca, sku, canal e uf. As demais
colunas ficam intactas. Os códigos anonimizados são sempre numéricos.
Processa em blocos (chunks) para suportar arquivos de 600MB+ sem carregar
tudo na memória de uma vez.
"""

import os
import time
import tempfile

import pandas as pd
import streamlit as st
import unicodedata

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CHUNK_SIZE = 200_000  # linhas processadas por bloco

COLUNAS_ALVO = {
    1: {"nome": "fornecedor", "aliases": ["fornecedor", "fabricante", "proveedor", "supplier", "manufacturer"]},
    2: {"nome": "ean", "aliases": ["ean", "codigo_ean", "codigo_barras", "codigo de barras"]},
    3: {"nome": "marca", "aliases": ["marca", "brand"]},
    4: {"nome": "sku", "aliases": ["sku", "produto", "descricao", "nome_produto", "nombre_sku"]},
    5: {"nome": "canal", "aliases": ["canal", "channel", "pdv_canal"]},
    6: {"nome": "uf", "aliases": ["uf", "estado", "state"]},
}
BASE_INDICE = 10**9  # até 999.999.999 valores únicos por coluna


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def normalizar(texto):
    texto = str(texto).strip().lower()
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


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
    """Casa colunas do arquivo com colunas-alvo pelo nome normalizado."""
    encontradas = {}
    for coluna in colunas_arquivo:
        col_norm = normalizar(coluna)
        for indice, info in COLUNAS_ALVO.items():
            if col_norm in [normalizar(a) for a in info["aliases"]]:
                encontradas[coluna] = indice
                break
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

    novos_registros = []
    for chave in set(chave_por_valor.values()):
        if chave not in mapa_coluna:
            codigo = indice * BASE_INDICE + proximo_seq[indice]
            mapa_coluna[chave] = codigo
            proximo_seq[indice] += 1
            novos_registros.append({"coluna": nome_coluna, "valor_original": chave, "codigo": codigo})

    codigo_por_valor = {v: mapa_coluna[chave] for v, chave in chave_por_valor.items()}
    resultado = serie.map(codigo_por_valor)

    # .map() força a coluna inteira para float64 quando há NaN misturado
    # com os códigos inteiros — por isso reconstruímos como object.
    resultado = pd.Series(
        [None if pd.isna(x) else int(x) for x in resultado],
        index=resultado.index, dtype="object"
    )
    resultado = resultado.where(resultado.notna(), serie)

    return resultado, novos_registros


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
    linhas = [
        {"coluna": COLUNAS_ALVO[indice]["nome"], "valor_original": chave, "codigo": codigo}
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


# ---------------------------------------------------------------------------
# Processamento: anonimização
# ---------------------------------------------------------------------------
def processar_anonimizacao(arquivo, sep, encoding, arquivo_mapa_anterior):
    tamanho_total = arquivo.size
    mapas, proximo_seq = carregar_mapa(arquivo_mapa_anterior)

    arquivo.seek(0)
    amostra = pd.read_csv(arquivo, sep=sep, encoding=encoding, nrows=1000, dtype=str)
    colunas_alvo = identificar_colunas(amostra.columns)
    arquivo.seek(0)

    if not colunas_alvo:
        st.warning("Nenhuma coluna-alvo (fornecedor, ean, marca, sku, canal, uf) foi encontrada no arquivo.")
        return

    st.info(
        "Colunas identificadas: "
        + ", ".join(f"**{col}** → {COLUNAS_ALVO[i]['nome']}" for col, i in colunas_alvo.items())
    )

    status = st.empty()
    barra = st.progress(0.0)
    inicio = time.time()

    saida_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    linhas_processadas = 0
    primeiro_chunk = True

    leitor = pd.read_csv(
        arquivo, sep=sep, encoding=encoding, chunksize=CHUNK_SIZE,
        dtype=str, keep_default_na=False
    )

    for chunk in leitor:
        for coluna, indice in colunas_alvo.items():
            chunk[coluna], _ = anonimizar_coluna(chunk[coluna], indice, mapas, proximo_seq)

        chunk.to_csv(saida_tmp, sep=sep, index=False, header=primeiro_chunk)
        primeiro_chunk = False
        linhas_processadas += len(chunk)

        fracao = min(arquivo.tell() / tamanho_total, 1.0) if tamanho_total else 0.0
        barra.progress(fracao)
        elapsed = time.time() - inicio
        eta_txt = formatar_tempo(elapsed / fracao - elapsed) if fracao > 0.02 else "calculando..."
        status.text(f"Anonimizando... {fracao * 100:.1f}% — {linhas_processadas:,} linhas — tempo restante: {eta_txt}")

    saida_tmp.close()
    barra.progress(1.0)
    status.text(f"Concluído: {linhas_processadas:,} linhas em {formatar_tempo(time.time() - inicio)}.")

    df_mapa = construir_df_mapa(mapas)

    st.session_state["resultado_anon"] = {
        "csv_path": saida_tmp.name,
        "mapa_csv_bytes": df_mapa.to_csv(index=False).encode("utf-8-sig"),
        "total_linhas": linhas_processadas,
        "total_mapeados": len(df_mapa),
    }


# ---------------------------------------------------------------------------
# Processamento: desanonimização
# ---------------------------------------------------------------------------
def processar_desanonimizacao(arquivo, sep, encoding, arquivo_mapa):
    if arquivo_mapa is None:
        st.warning("Envie o arquivo de mapeamento (referência) para desanonimizar.")
        return

    tamanho_total = arquivo.size
    df_mapa = pd.read_csv(arquivo_mapa, dtype=str)
    mapa_reverso = {
        coluna: dict(zip(grupo["codigo"].astype(str), grupo["valor_original"]))
        for coluna, grupo in df_mapa.groupby("coluna")
    }

    arquivo.seek(0)
    amostra = pd.read_csv(arquivo, sep=sep, encoding=encoding, nrows=1000, dtype=str)
    colunas_no_arquivo = identificar_colunas(amostra.columns)
    colunas_para_restaurar = {
        col: COLUNAS_ALVO[idx]["nome"]
        for col, idx in colunas_no_arquivo.items()
        if COLUNAS_ALVO[idx]["nome"] in mapa_reverso
    }
    arquivo.seek(0)

    if not colunas_para_restaurar:
        st.warning("Nenhuma coluna do arquivo bate com o mapeamento enviado.")
        return

    st.info("Colunas a restaurar: " + ", ".join(f"**{c}**" for c in colunas_para_restaurar))

    status = st.empty()
    barra = st.progress(0.0)
    inicio = time.time()

    saida_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    linhas_processadas = 0
    primeiro_chunk = True

    leitor = pd.read_csv(
        arquivo, sep=sep, encoding=encoding, chunksize=CHUNK_SIZE,
        dtype=str, keep_default_na=False
    )

    for chunk in leitor:
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
# Interface
# ---------------------------------------------------------------------------
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 26px;
        background-color: #E8ECF3;
        color: #0B1F3A;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0B1F3A;
        box-shadow: 0 2px 10px rgba(11, 31, 58, 0.3);
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
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
        right: 60rem;
        z-index: 1000;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem;
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
    st.image("logo_scanntech.png", width=220)

st.markdown(
    "<h1 style='color:#0B1F3A; margin-bottom:0;'>Anonimizador de Dados</h1>"
    "<p style='color:#5A6B85; margin-top:2px;'>Fornecedor · EAN · Marca · SKU · Canal · UF"
    " — códigos sempre numéricos</p>",
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    separador = st.selectbox("Separador do CSV", [";", ",", "\t", "|"], index=0)
    encoding = st.selectbox("Codificação", ["utf-8-sig", "utf-8", "latin1", "cp1252"], index=0)

aba_anonimizar, aba_desanonimizar = st.tabs(["🔒 Anonimizar", "🔓 Desanonimizar"])

with aba_anonimizar:
    with st.container(border=True):
        arquivo = st.file_uploader("Arquivo CSV para anonimizar", type=["csv", "txt"], key="upload_anon")

        if arquivo is not None:
            st.caption(f"📄 {arquivo.name} — {arquivo.size / 1_048_576:.1f} MB")
            if st.button("▶️ Iniciar anonimização", type="primary", key="btn_anon"):
                processar_anonimizacao(arquivo, separador, encoding, None)

    if "resultado_anon" in st.session_state:
        resultado = st.session_state["resultado_anon"]
        st.write("")
        with st.container(border=True):
            st.success("✅ Anonimização concluída")
            c1, c2 = st.columns(2)
            c1.metric("Linhas processadas", f"{resultado['total_linhas']:,}")
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
        arquivo_desanon = st.file_uploader("CSV anonimizado", type=["csv", "txt"], key="upload_desanon")

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