# Anonimizador de Dados — Scanntech

Anonimiza `fornecedor`, `ean`, `marca`, `sku`, `canal` e `uf`. As demais
colunas ficam intactas. Os códigos gerados são sempre numéricos.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run anonimizador.py
```

Abre em `http://localhost:8501`.

## Uso

1. Selecione o separador e a codificação do seu CSV na barra lateral (padrão: `;` e `utf-8-sig`, comum em exportações brasileiras).
2. (Opcional) Envie um mapeamento anterior para manter os mesmos códigos entre execuções.
3. Envie o arquivo CSV (aceita 600MB+ — o limite está configurado em `.streamlit/config.toml`).
4. Clique em "Iniciar anonimização" e acompanhe a barra de progresso com estimativa de tempo.
5. Baixe o CSV anonimizado e o mapeamento (referência) ao final.

## Como funciona o código anonimizado

Cada coluna tem um prefixo fixo: `fornecedor`→1, `ean`→2, `marca`→3, `sku`→4,
`canal`→5, `uf`→6. O código final é `indice * 1.000.000.000 + sequencial`
(ex: `2000000001`). Isso garante que nunca colide entre colunas, mesmo sendo
100% numérico — sem letras, sem hash.

## Limitações conhecidas

- O download final carrega o arquivo de saída inteiro na memória no momento
  do clique (limitação do `st.download_button`). Para 600MB isso é normal,
  mas exige RAM disponível equivalente ao tamanho do arquivo nesse instante.
- O mapeamento não persiste automaticamente entre sessões — se quiser manter
  os mesmos códigos ao longo do tempo, baixe o mapeamento e reenvie-o na
  próxima execução.
- Não desanonimiza (não é o objetivo desta versão). Se precisar, dá para
  adicionar um script simples de leitura reversa usando o próprio mapeamento.
