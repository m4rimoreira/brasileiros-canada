"""
Brasileiros no Canadá — Análise Histórica (1980–2024)
------------------------------------------------------
Fonte: Kaggle — "Immigration to Canada by Country of Origin"
Dados oficiais do IRCC (Immigration, Refugees and Citizenship Canada)
via open.canada.ca, consolidados de 1980 a 2024.

Perguntas respondidas:
1. Evolução da imigração brasileira para o Canadá (1980–2024)
2. Comparação com Argentina, Colômbia, Chile e Peru
3. Identificação de picos e quedas históricas

Autora: Mariana Moreira
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

PASTA_OUTPUT = "outputs"
os.makedirs(PASTA_OUTPUT, exist_ok=True)

plt.style.use("dark_background")
CORES = {
    "brasil": "#a78bfa",
    "argentina": "#f472b6",
    "colombia": "#67e8f9",
    "chile": "#c4b5fd",
    "peru": "#64748b",
    "texto": "#f5f3ff",
    "grid": "#292d42",
}


def baixar_dados():
    """Baixa o dataset do Kaggle via kagglehub."""
    print("⏳ Baixando dataset do Kaggle...")
    path = kagglehub.dataset_download(
        "aliranjipour/immigration-to-canada-by-country-of-origin"
    )
    print(f"✔ Dataset baixado em: {path}")
    return path


def carregar_dados(path):
    """
    Carrega o CSV e transforma de formato 'wide' (anos como colunas)
    para 'long' (uma linha por país-ano), de forma robusta:
    detecta automaticamente a coluna de país e as colunas de ano.
    """

    arquivos = glob.glob(os.path.join(path, "*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {path}")
    arquivo = arquivos[0]
    print(f"📄 Lendo: {arquivo}")

    df = pd.read_csv(arquivo)


    print("🔎 Colunas encontradas:", list(df.columns)[:10], "...")


    coluna_pais = None
    for col in df.columns:
        if not str(col).strip().isdigit():
            coluna_pais = col
            break
    if coluna_pais is None:
        raise ValueError("Não foi possível identificar a coluna de país.")


    colunas_anos = [c for c in df.columns if str(c).strip().isdigit()]
    colunas_anos = sorted(colunas_anos, key=lambda x: int(str(x).strip()))

    if not colunas_anos:
        raise ValueError("Nenhuma coluna de ano encontrada no CSV.")


    df = df[[coluna_pais] + colunas_anos].copy()
    df = df.rename(columns={coluna_pais: "pais"})


    df_long = df.melt(
        id_vars="pais",
        value_vars=colunas_anos,
        var_name="ano",
        value_name="imigrantes"
    )
    df_long["ano"] = df_long["ano"].astype(str).str.strip().astype(int)
    df_long["imigrantes"] = pd.to_numeric(
        df_long["imigrantes"], errors="coerce"
    ).fillna(0)

    agregados = [
        "World", "Africa", "Asia", "Europe", "North America",
        "South America", "Oceania", "European Union", "Not stated",
        "Unknown", "Total"
    ]
    df_long = df_long[~df_long["pais"].isin(agregados)]

    return df_long


def grafico_brasil(df):
    br = df[df["pais"] == "Brazil"].sort_values("ano")

    if br.empty:
        print("⚠ Nenhum dado encontrado para 'Brazil'. Verifique o nome do país no CSV.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(br["ano"], br["imigrantes"],
            color=CORES["brasil"], linewidth=2.5, marker="o", markersize=3)
    ax.fill_between(br["ano"], br["imigrantes"],
                    color=CORES["brasil"], alpha=0.15)


    pico = br.loc[br["imigrantes"].idxmax()]
    ax.annotate(
        f"Pico: {pico['imigrantes']:,.0f}\n({pico['ano']:.0f})",
        xy=(pico["ano"], pico["imigrantes"]),
        xytext=(pico["ano"] - 8, pico["imigrantes"] * 1.15),
        color=CORES["texto"], fontsize=10,
        arrowprops=dict(arrowstyle="->", color=CORES["brasil"])
    )

    ax.set_title("Imigração brasileira para o Canadá (1980–2024)",
                 fontsize=14, color=CORES["texto"], pad=15)
    ax.set_xlabel("Ano", color=CORES["texto"])
    ax.set_ylabel("Admissões de residentes permanentes", color=CORES["texto"])
    ax.grid(alpha=0.2, color=CORES["grid"])

    plt.tight_layout()
    caminho = f"{PASTA_OUTPUT}/grafico_brasil_evolucao.png"
    plt.savefig(caminho, dpi=150, facecolor="#0f1117")
    plt.close()
    print(f"✔ Gráfico salvo: {caminho}")


def grafico_comparacao(df):
    paises = {
        "Brazil": ("Brasil", CORES["brasil"]),
        "Argentina": ("Argentina", CORES["argentina"]),
        "Colombia": ("Colômbia", CORES["colombia"]),
        "Chile": ("Chile", CORES["chile"]),
        "Peru": ("Peru", CORES["peru"]),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    algum_dado = False

    for pais_en, (pais_pt, cor) in paises.items():
        dados = df[df["pais"] == pais_en].sort_values("ano")
        if not dados.empty:
            ax.plot(dados["ano"], dados["imigrantes"],
                    label=pais_pt, color=cor, linewidth=2)
            algum_dado = True

    if not algum_dado:
        print("⚠ Nenhum país sul-americano encontrado no CSV.")
        plt.close()
        return

    ax.set_title("Imigração sul-americana para o Canadá (1980–2024)",
                 fontsize=14, color=CORES["texto"], pad=15)
    ax.set_xlabel("Ano", color=CORES["texto"])
    ax.set_ylabel("Admissões de residentes permanentes", color=CORES["texto"])
    ax.legend(loc="upper left", frameon=False, labelcolor=CORES["texto"])
    ax.grid(alpha=0.2, color=CORES["grid"])

    plt.tight_layout()
    caminho = f"{PASTA_OUTPUT}/grafico_comparacao_sulamerica.png"
    plt.savefig(caminho, dpi=150, facecolor="#0f1117")
    plt.close()
    print(f"✔ Gráfico salvo: {caminho}")


def grafico_ranking(df):
    paises_sulamerica = [
        "Brazil", "Argentina", "Colombia", "Chile", "Peru",
        "Venezuela", "Ecuador", "Bolivia", "Paraguay", "Uruguay"
    ]
    dados = df[df["pais"].isin(paises_sulamerica)]
    totais = dados.groupby("pais")["imigrantes"].sum().sort_values(ascending=True)

    if totais.empty:
        print("⚠ Nenhum dado sul-americano encontrado para o ranking.")
        return


    traducao = {
        "Brazil": "Brasil", "Argentina": "Argentina", "Colombia": "Colômbia",
        "Chile": "Chile", "Peru": "Peru", "Venezuela": "Venezuela",
        "Ecuador": "Equador", "Bolivia": "Bolívia", "Paraguay": "Paraguai",
        "Uruguay": "Uruguai"
    }
    totais.index = [traducao.get(p, p) for p in totais.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    cores_barras = [CORES["brasil"] if p == "Brasil" else CORES["peru"]
                    for p in totais.index]
    bars = ax.barh(totais.index, totais.values, color=cores_barras, alpha=0.85)

    for bar, val in zip(bars, totais.values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}", va="center", color=CORES["texto"], fontsize=9)

    ax.set_title("Total de imigrantes para o Canadá por país sul-americano (1980–2024)",
                 fontsize=13, color=CORES["texto"], pad=15)
    ax.set_xlabel("Total de admissões", color=CORES["texto"])
    ax.grid(axis="x", alpha=0.2, color=CORES["grid"])

    plt.tight_layout()
    caminho = f"{PASTA_OUTPUT}/grafico_ranking_sulamerica.png"
    plt.savefig(caminho, dpi=150, facecolor="#0f1117")
    plt.close()
    print(f"✔ Gráfico salvo: {caminho}")


def imprimir_insights(df):
    br = df[df["pais"] == "Brazil"].sort_values("ano")

    if br.empty:
        print("\n⚠ Não foi possível gerar insights: dados do Brasil não encontrados.")
        return

    total_br = br["imigrantes"].sum()
    pico = br.loc[br["imigrantes"].idxmax()]
    ultimo_ano = br.iloc[-1]

    print("\n📊 INSIGHTS — BRASILEIROS NO CANADÁ (1980–2024)")
    print("=" * 55)
    print(f"• Total de admissões no período: {total_br:,.0f}")
    print(f"• Pico histórico: {pico['imigrantes']:,.0f} em {pico['ano']:.0f}")
    print(f"• Último ano ({ultimo_ano['ano']:.0f}): {ultimo_ano['imigrantes']:,.0f} admissões")
    print(f"• Média anual: {br['imigrantes'].mean():,.0f}")
    print("=" * 55)


if __name__ == "__main__":
    path = baixar_dados()
    df = carregar_dados(path)
    print(f"✔ Dados carregados: {len(df):,} registros país-ano\n")

 
    possiveis_brasil = df[df["pais"].astype(str).str.contains("Bra", case=False, na=False)]["pais"].unique()
    if len(possiveis_brasil) > 0:
        print(f"🔎 Países com 'Bra' no nome: {list(possiveis_brasil)}\n")

    grafico_brasil(df)
    grafico_comparacao(df)
    grafico_ranking(df)
    imprimir_insights(df)

    print("\n✅ Análise concluída! Veja a pasta 'outputs/'.")