import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
import pydeck as pdk
import plotly.express as px
from streamlit_plotly_events import plotly_events

# --- Configurações iniciais ---
st.set_page_config(page_title="Dashboard Banco Inter", layout="wide")


# --- Funções de carga e geocodificação ---
@st.cache_data
def load_data(path):
    return pd.read_excel(path)


@st.cache_data
def geocode_comarcas(comarcas):
    geolocator = Nominatim(user_agent="dashboard_app")
    records = []

    # Dicionário de exceções com geocodificação forçada
    excecoes = {
        "brasília": "Brasília, DF, Brasil",
        "são paulo": "São Paulo, SP, Brasil"
    }

    for comarca in comarcas.dropna().unique():
        try:
            comarca_normalizada = comarca.strip().lower()

            # Verifica se a comarca está no dicionário de exceções
            if comarca_normalizada in excecoes:
                query = excecoes[comarca_normalizada]
            else:
                query = f"{comarca}, Brasil"

            loc = geolocator.geocode(query)
            if loc:
                records.append({"Comarca": comarca, "lat": loc.latitude, "lon": loc.longitude})
        except:
            continue

    return pd.DataFrame(records)


# --- Carrega e processa dados ---
EXCEL_PATH = "Base de Processos VLF.xlsx"
df = load_data(EXCEL_PATH)
coords = geocode_comarcas(df['Comarca'])
df_full = pd.merge(df, coords, on='Comarca', how='inner')

# Converte datas para cálculos
dist_col = 'Data da Distribuição'
last_mov_col = 'Data da Última Movimentação'
dlq_col = 'Liquidação Inicial Sem Dedução'
df_full[dist_col] = pd.to_datetime(df_full[dist_col], dayfirst=True, errors='coerce')
df_full[last_mov_col] = pd.to_datetime(df_full[last_mov_col], dayfirst=True, errors='coerce')

# --- Cálculos de KPI ---
mask_closed = df_full['Transito em Julgado'].str.lower().str.strip() == 'sim'
df_full.loc[mask_closed, 'Duration'] = (
        df_full.loc[mask_closed, last_mov_col] - df_full.loc[mask_closed, dist_col]).dt.days
avg_duration = int(df_full.loc[mask_closed, 'Duration'].dropna().mean()) if mask_closed.any() else 0

mask_active = df_full['Transito em Julgado'].str.lower().str.strip() == 'não'
risk_current = df_full.loc[mask_active, dlq_col].sum()

econ_col = 'Economia Concreta'
dep_col = 'Total Garantido'
agreement_col = 'Acordo'
econ_total = df_full[econ_col].fillna(0).sum()
dep_total = df_full[dep_col].fillna(0).sum()
num_acordos = df_full[agreement_col].str.lower().str.strip().eq('sim').sum()

# --- Agrega dados para mapa e por comarca ---
counts = df_full.groupby('Comarca').size().reset_index(name='Processos')
map_df = pd.merge(coords, counts, on='Comarca', how='inner')

# --- Cabeçalho com logos ---
col_logo1, col_spacer, col_logo2 = st.columns([1, 8, 1])
with col_logo1:
    st.image("logo_banco.png", width=200)
with col_logo2:
    st.image("logo_escritorio.png", width=200)

# --- Título principal ---
st.markdown(
    "<h1 style='text-align:center; color:#FF6600; margin:10px 0;'>Panorama da Carteira do VLF - Banco Inter</h1>",
    unsafe_allow_html=True
)

# --- Cria abas ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Análise de Objetos",
    "Análise de Terceirizadas",
    "Análise Inter Pag",
    "Análise de Acordos",
    "Cláusula 11ª – CCT",
    "Êxito"
])

# --- Aba Dashboard ---
with tab1:
    # Linha 1 de KPIs
    total_processos = len(df_full)
    num_comarcas = counts['Comarca'].nunique()
    idx_max = counts['Processos'].idxmax()
    comarca_max = counts.loc[idx_max, 'Comarca']
    max_processos = counts.loc[idx_max, 'Processos']
    ativos = df_full['Transito em Julgado'].str.strip().str.lower() != 'sim'
    num_ativos = ativos.sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Processos", f"{total_processos}")
    k2.metric("Processos Ativos", f"{num_ativos}")
    k3.metric("Comarcas com Processos", f"{num_comarcas}")
    k4.metric("Maior Concentração", f"{comarca_max} ({max_processos})")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Proveito Econômico (R$)", f"{econ_total:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    k6.metric("Total Garantido (R$)", f"{dep_total:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    k7.metric("Acordos Firmados", f"{num_acordos}")

    # Cálculo correto da Economia CCT apenas com processos ativos e com dedução válida
    df_cct = df_full[df_full['Transito em Julgado'].str.strip().str.lower() != 'sim'].copy()
    df_cct['Deduções CCT'] = df_cct.apply(
        lambda row: row['Liquidação Inicial Sem Dedução'] - row['Liquidação Inicial Com Dedução']
        if row['Liquidação Inicial Com Dedução'] > 0 else 0,
        axis=1
    )
    economia_cct = df_cct['Deduções CCT'].sum()
    k8.metric("Economia CCT (R$)", f"{economia_cct:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Distribuição Geográfica</h3>", unsafe_allow_html=True)

    col_map, col_ranking = st.columns([3, 2])
    with col_map:
        show_heatmap = st.toggle("Exibir Heatmap", key="tab1_heatmap")

        view = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5, pitch=0)

        layers = [
            pdk.Layer(
                "GeoJsonLayer",
                data="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                stroked=True,
                filled=False,
                get_line_color=[100, 100, 100],
                line_width_min_pixels=1
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_fill_color=[255, 102, 0, 220],
                get_radius=20000,
                pickable=True,
                auto_highlight=True
            ),
            pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_text="Processos",
                get_color=[50, 50, 50],
                get_size=12,
                get_angle=0,
                get_text_anchor="middle"
            )
        ]

        if show_heatmap:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=map_df,
                    get_position=["lon", "lat"],
                    get_weight="Processos",  # peso baseado na quantidade de processos
                    aggregation="SUM"
                )
            )

        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=view,
            layers=layers,
            tooltip={"text": "Comarca: {Comarca}\nProcessos: {Processos}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

    with col_ranking:
        st.markdown("<h3 style='color:#201747'>Top 10 Comarcas</h3>", unsafe_allow_html=True)
        top10 = counts.nlargest(10, 'Processos')
        fig1 = px.bar(top10, x='Processos', y='Comarca', orientation='h', color_discrete_sequence=["#FF6600"])
        fig1.update_yaxes(categoryorder='array', categoryarray=top10['Comarca'][::-1].tolist())
        fig1.update_layout(
            title_text='',  # Remove o título do gráfico
            title_x=0.5,
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF'
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Processos por Comarca</h3>", unsafe_allow_html=True)
    comarcas_list = counts.sort_values(by='Processos', ascending=False)['Comarca']
    selected = st.selectbox("Selecione uma Comarca", comarcas_list)
    df_sel = df_full[df_full['Comarca'] == selected][
        ['Nº processo principal', 'Adverso principal', 'Empregado Próprio']].copy()
    df_sel.index = range(1, len(df_sel) + 1)
    st.dataframe(df_sel, use_container_width=True, height=300)

    st.markdown("---")
    # --- Gráfico de Pizza: Tipos de Vínculo ---
    st.markdown("### Distribuição Geral por Tipo de Empregado")

    df_vinculo = df_full.copy()
    df_vinculo['Empregado Próprio'] = df_vinculo['Empregado Próprio'].astype(str).str.strip().str.lower()
    df_vinculo['Terceirizada / Inter Pag'] = df_vinculo['Terceirizada / Inter Pag'].astype(str).str.lower().fillna('')


    def classificar_vinculo(row):
        if row['Empregado Próprio'] == 'sim':
            return 'Empregado Próprio'
        elif 'inter pag' in row['Terceirizada / Inter Pag']:
            return 'Inter Pag'
        else:
            return 'Terceirizado'


    df_vinculo['Tipo de Empregado'] = df_vinculo.apply(classificar_vinculo, axis=1)

    dados_pizza = df_vinculo['Tipo de Empregado'].value_counts().reset_index()
    dados_pizza.columns = ['Categoria', 'Quantidade']

    fig_pizza_emp = px.pie(
        dados_pizza,
        values='Quantidade',
        names='Categoria',
        title='Distribuição Geral por Tipo de Empregado',
        color_discrete_sequence=px.colors.sequential.Oranges
    )

    fig_pizza_emp.update_traces(textinfo='label+percent', pull=[0.05] * len(dados_pizza))
    fig_pizza_emp.update_layout(width=800, height=600)

    st.plotly_chart(fig_pizza_emp, use_container_width=False)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Processos Distribuídos por Mês</h3>", unsafe_allow_html=True)

    ts = df_full.dropna(subset=[dist_col]).groupby(df_full[dist_col].dt.to_period('M')).size().reset_index(name='Count')
    ts[dist_col] = ts[dist_col].dt.to_timestamp()
    ts['Month'] = ts[dist_col].dt.strftime('%Y-%m')

    fig2 = px.bar(
        ts,
        x='Month',
        y='Count',
        color_discrete_sequence=["#FF6600"],
        labels={'Count': 'Processos'}
    )

    fig2.update_layout(
        title_text='',  # Remove o título automático
        title_x=0.5,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Evolução da Carteira Ativa</h3>", unsafe_allow_html=True)

    # Entradas: +1 na data de distribuição
    entradas = df_full.dropna(subset=[dist_col]).copy()
    entradas['Data'] = entradas[dist_col]
    entradas['Delta'] = 1

    # Saídas: -1 na data da última movimentação (apenas com trânsito em julgado)
    saidas = df_full[df_full['Transito em Julgado'].str.strip().str.lower() == 'sim'].copy()
    saidas = saidas.dropna(subset=[last_mov_col])
    saidas['Data'] = saidas[last_mov_col]
    saidas['Delta'] = -1

    # Combina entradas e saídas
    fluxo = pd.concat([entradas[['Data', 'Delta']], saidas[['Data', 'Delta']]])
    fluxo = fluxo.dropna(subset=['Data'])
    fluxo = fluxo.groupby('Data').sum().sort_index()
    fluxo['Processos Ativos'] = fluxo['Delta'].cumsum()
    fluxo = fluxo.reset_index()

    # Converte para mês (agregação final)
    fluxo['Mês'] = fluxo['Data'].dt.to_period('M').dt.to_timestamp()
    mensal = fluxo.groupby('Mês')['Processos Ativos'].max().reset_index()

    # Gráfico final
    fig_carteira = px.line(
        mensal,
        x='Mês',
        y='Processos Ativos',
        markers=True,
        title="Evolução da Carteira (Processos Ativos)",
        labels={'Mês': 'Mês', 'Processos Ativos': 'Total Acumulado'},
        color_discrete_sequence=["#FF6600"]
    )
    fig_carteira.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    st.plotly_chart(fig_carteira, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Evolução do Risco</h3>", unsafe_allow_html=True)

    # Entradas de risco: valor no dia da distribuição
    entradas_risco = df_full.dropna(subset=[dist_col, dlq_col]).copy()
    entradas_risco['Data'] = entradas_risco[dist_col]
    entradas_risco['Delta'] = entradas_risco[dlq_col]

    # Saídas de risco: -valor no dia da última movimentação (se houve trânsito em julgado)
    saidas_risco = df_full[df_full['Transito em Julgado'].str.strip().str.lower() == 'sim'].copy()
    saidas_risco = saidas_risco.dropna(subset=[last_mov_col, dlq_col])
    saidas_risco['Data'] = saidas_risco[last_mov_col]
    saidas_risco['Delta'] = -saidas_risco[dlq_col]

    # Combina
    fluxo_risco = pd.concat([entradas_risco[['Data', 'Delta']], saidas_risco[['Data', 'Delta']]])
    fluxo_risco = fluxo_risco.dropna(subset=['Data'])
    fluxo_risco = fluxo_risco.groupby('Data').sum().sort_index()
    fluxo_risco['Risco Exposto'] = fluxo_risco['Delta'].cumsum()
    fluxo_risco = fluxo_risco.reset_index()

    # Agrega por mês
    fluxo_risco['Mês'] = fluxo_risco['Data'].dt.to_period('M').dt.to_timestamp()
    mensal_risco = fluxo_risco.groupby('Mês')['Risco Exposto'].max().reset_index()
    mensal_risco['Média Móvel 3M'] = mensal_risco['Risco Exposto'].rolling(3, min_periods=1).mean()

    # Gráfico
    fig_risk = px.bar(
        mensal_risco,
        x='Mês',
        y='Risco Exposto',
        color_discrete_sequence=["#FF6600"],
        labels={'Risco Exposto': 'R$', 'Mês': 'Mês'},
        title="Evolução do Risco Mensal"
    )
    fig_risk.add_scatter(
        x=mensal_risco['Mês'],
        y=mensal_risco['Média Móvel 3M'],
        mode='lines',
        name='Média Móvel 3M'
    )
    fig_risk.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    st.plotly_chart(fig_risk, use_container_width=True)

# --- Aba Análise de Objetos ---
with tab2:
    st.markdown("<h2 style='color:#201747'>Análise de Objetos dos Pedidos</h2>", unsafe_allow_html=True)

    # Filtra apenas empregados próprios
    df_proprio = df_full[df_full['Empregado Próprio'].str.strip().str.lower() == 'sim']

    obj_cols = [c for c in df_full.columns if 'objeto' in c.lower()]
    if obj_cols:
        obj_col = obj_cols[0]
        s = df_proprio[obj_col].dropna().str.split('|').explode().str.strip()
        obj_counts = s.value_counts().reset_index()
        obj_counts.columns = ['Objeto', 'Incidência']

        top_objs = obj_counts.head(10)
        fig_obj = px.bar(
            top_objs,
            x='Incidência',
            y='Objeto',
            orientation='h',
            color_discrete_sequence=["#FF6600"]
        )
        fig_obj.update_yaxes(categoryorder='array', categoryarray=top_objs['Objeto'][::-1].tolist())
        fig_obj.update_layout(
            title_text='',  # Remove título redundante
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_obj, use_container_width=True)

        st.markdown("<h4 style='color:#201747'>Filtrar por Objeto</h4>", unsafe_allow_html=True)

        # Novo filtro por tipo de empregado
        tipo_empregado = st.radio("Tipo de empregado:", ["Todos", "Próprio", "Terceirizado"], horizontal=True)

        if st.checkbox("Selecionar todos", value=False):
            sel_objs = obj_counts['Objeto'].tolist()
        else:
            sel_objs = st.multiselect(
                "Selecionar objetos específicos",
                obj_counts['Objeto'].tolist(),
                default=obj_counts['Objeto'].head(3).tolist()
            )

        if sel_objs:
            # Aplica filtro por tipo de empregado
            if tipo_empregado == "Próprio":
                df_filtrado = df_full[df_full['Empregado Próprio'].str.strip().str.lower() == 'sim'].copy()
            elif tipo_empregado == "Terceirizado":
                df_filtrado = df_full[df_full['Empregado Próprio'].str.strip().str.lower() == 'não'].copy()
            else:
                df_filtrado = df_full.copy()

            # Filtro por objetos
            mask = pd.Series(True, index=df_filtrado.index)
            for o in sel_objs:
                mask &= df_filtrado[obj_col].str.contains(o, na=False)
            df_obj = df_filtrado[mask]

            cols_show = ['Nº processo principal', 'Adverso principal', 'Empregado Próprio', obj_col]
            st.markdown(
                f"<h4 style='color:#201747'>Processos com objetos: {', '.join(sel_objs)}</h4>",
                unsafe_allow_html=True
            )
            st.dataframe(df_obj[cols_show].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("<h2 style='color:#201747'>Radar de Objetos em Casos Procedentes e Parcialmente Procedentes</h2>",
                unsafe_allow_html=True)

    procedentes_df = df_full[
        df_full['Procedência Atual'].str.strip().str.lower().isin(['procedente', 'parcialmente procedente'])
    ]
    obj_cols_procs = [c for c in procedentes_df.columns if 'objeto' in c.lower()]
    if obj_cols_procs:
        obj_col = obj_cols_procs[0]
        s_proc = procedentes_df[obj_col].dropna().str.split('|').explode().str.strip()
        obj_counts_proc = s_proc.value_counts().reset_index()
        obj_counts_proc.columns = ['Objeto', 'Count']

        fig_radar = px.line_polar(
            obj_counts_proc.head(10),
            r='Count',
            theta='Objeto',
            line_close=True,
            title='Principais Objetos nos Casos Procedentes / Parcialmente Procedentes',
            color_discrete_sequence=["#FF6600"]
        )
        fig_radar.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            font=dict(family="Open Sans", size=14)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "Os objetos representados neste gráfico foram os mais frequentes em sentenças procedentes ou parcialmente procedentes.")
    else:
        st.info("Coluna de objeto não encontrada para gerar o radar.")

# --- Aba Análise de Terceirizadas ---
with tab3:
    st.markdown("# Análise de Processos Terceirizados")

    # Filtra processos que não são de empregado próprio E não são Inter Pag
    terceiro_df = df_full[
        (df_full['Empregado Próprio'].str.strip().str.lower() != 'sim') &
        (~df_full['Terceirizada / Inter Pag'].str.lower().str.contains('inter pag', na=False))
        ].copy()

    st.metric("Processos Empregados Terceiros", len(terceiro_df))

    # Top 10 comarcas
    terc_counts = terceiro_df.groupby('Comarca').size().reset_index(name='Processos')
    terc_top = terc_counts.nlargest(10, 'Processos')
    st.markdown("## Top 10 Comarcas (Empregados Terceiros)")
    fig_terc = px.bar(terc_top, x='Processos', y='Comarca', orientation='h', color_discrete_sequence=["#FF6600"])
    fig_terc.update_yaxes(categoryorder='array', categoryarray=terc_top['Comarca'][::-1].tolist())
    st.plotly_chart(fig_terc, use_container_width=True)

    # Lista de processos
    st.markdown("## Lista de Processos Empregados Terceiros")

    # Mapeamento explícito dos nomes exatos das colunas
    df_exibe = terceiro_df[[
        'Nº processo principal',
        'Comarca',
        'Adverso principal',
        'Terceirizada / Inter Pag'
    ]].copy()

    df_exibe.columns = [
        'Número do Processo',
        'Comarca',
        'Reclamante',
        'Empresa Terceirizada'
    ]

    st.dataframe(df_exibe.reset_index(drop=True), use_container_width=True, height=300)

    # Empresas terceirizadas
    s_emp = terceiro_df['Terceirizada / Inter Pag'].dropna().str.split('|').explode().str.strip()
    s_emp = s_emp[s_emp.str.lower() != 'inter pag']
    emp_counts = s_emp.value_counts().reset_index()
    emp_counts.columns = ['Empresa', 'Count']
    st.markdown("## Principais Empresas Terceirizadas (Top 10)")
    fig_emp = px.bar(emp_counts.head(10), x='Count', y='Empresa', orientation='h', color_discrete_sequence=["#FF6600"])
    fig_emp.update_yaxes(categoryorder='array', categoryarray=emp_counts['Empresa'].head(10)[::-1].tolist())
    st.plotly_chart(fig_emp, use_container_width=True)

    # Evolução temporal dos processos por terceirizada
    st.markdown("## Evolução no número de Processos por Terceirizada")

    colunas_necessarias = [
        'Data da Distribuição',
        'Terceirizada / Inter Pag',
        'Transito em Julgado',
        'Data da Última Movimentação'
    ]
    colunas_existentes = [col for col in colunas_necessarias if col in terceiro_df.columns]

    if len(colunas_existentes) < len(colunas_necessarias):
        st.warning("Não foi possível gerar o gráfico de evolução. As seguintes colunas estão ausentes: "
                   + ", ".join(set(colunas_necessarias) - set(colunas_existentes)))
    else:
        evol_df = terceiro_df[colunas_necessarias].copy()
        evol_df = evol_df[~evol_df['Terceirizada / Inter Pag'].str.lower().str.contains('inter pag', na=False)]

        evol_df['Terceirizada / Inter Pag'] = evol_df['Terceirizada / Inter Pag'].str.split('|')
        evol_df = evol_df.explode('Terceirizada / Inter Pag')
        evol_df['Terceirizada / Inter Pag'] = evol_df['Terceirizada / Inter Pag'].str.strip()

        # Cria DataFrame com +1 na distribuição
        ativos = evol_df[['Data da Distribuição', 'Terceirizada / Inter Pag']].copy()
        ativos['Data'] = pd.to_datetime(ativos['Data da Distribuição'], errors='coerce')
        ativos['Contagem'] = 1

        # Cria DataFrame com -1 na data de encerramento
        encerrados = evol_df[
            (evol_df['Transito em Julgado'].str.lower() == 'sim') & evol_df['Data da Última Movimentação'].notna()]
        encerrados['Data'] = pd.to_datetime(encerrados['Data da Última Movimentação'], errors='coerce')
        encerrados = encerrados[['Data', 'Terceirizada / Inter Pag']]
        encerrados['Contagem'] = -1

        # Une, soma e acumula
        linha_df = pd.concat([ativos[['Data', 'Terceirizada / Inter Pag', 'Contagem']], encerrados], ignore_index=True)
        linha_df = linha_df.dropna(subset=['Data'])
        linha_df = linha_df.sort_values('Data')
        linha_df = linha_df.groupby(['Data', 'Terceirizada / Inter Pag']).sum().reset_index()
        linha_df['Ativos'] = linha_df.groupby('Terceirizada / Inter Pag')['Contagem'].cumsum()

        # Filtro por empresas mais relevantes
        principais = linha_df['Terceirizada / Inter Pag'].value_counts().head(5).index.tolist()
        selecao = st.multiselect("Selecione as terceirizadas para visualizar",
                                 linha_df['Terceirizada / Inter Pag'].unique().tolist(), default=principais)

        linha_filtrada = linha_df[linha_df['Terceirizada / Inter Pag'].isin(selecao)]

        linha_filtrada = linha_filtrada.rename(columns={'Terceirizada / Inter Pag': 'Terceirizada'})

        fig_linha = px.line(
            linha_filtrada,
            x='Data',
            y='Ativos',
            color='Terceirizada',
            hover_name='Terceirizada',
            labels={'Ativos': 'Processos Ativos', 'Data': 'Data'},
            title='Evolução de Processos Ativos por Terceirizada'
        )

        st.plotly_chart(fig_linha, use_container_width=True)

with tab4:
    st.markdown("# Análise de Processos Inter Pag")

    # Filtra processos de Inter Pag (terceirizados)
    interpag_df = df_full[
        (df_full['Empregado Próprio'].str.strip().str.lower() != 'sim') &
        (df_full['Terceirizada / Inter Pag'].str.lower().str.contains('inter pag', na=False))
        ].copy()

    st.metric("Processos Inter Pag", len(interpag_df))

    # Top 10 comarcas
    ip_counts = interpag_df.groupby('Comarca').size().reset_index(name='Processos')
    ip_top = ip_counts.nlargest(10, 'Processos')
    st.markdown("## Top 10 Comarcas (Inter Pag)")
    fig_ip = px.bar(ip_top, x='Processos', y='Comarca', orientation='h', color_discrete_sequence=["#FF6600"])
    fig_ip.update_yaxes(categoryorder='array', categoryarray=ip_top['Comarca'][::-1].tolist())
    st.plotly_chart(fig_ip, use_container_width=True)

    # Lista de processos
    st.markdown("## Lista de Processos Inter Pag")
    df_exibe = interpag_df[[
        'Nº processo principal',
        'Comarca',
        'Adverso principal',
        'Terceirizada / Inter Pag'
    ]].copy()

    df_exibe.columns = [
        'Número do Processo',
        'Comarca',
        'Reclamante',
        'Empresa Terceirizada'
    ]
    st.dataframe(df_exibe.reset_index(drop=True), use_container_width=True, height=300)

    # Evolução temporal dos processos Inter Pag
    st.markdown("## Evolução no número de Processos Inter Pag")

    colunas_necessarias = [
        'Data da Distribuição',
        'Terceirizada / Inter Pag',
        'Transito em Julgado',
        'Data da Última Movimentação'
    ]
    colunas_existentes = [col for col in colunas_necessarias if col in interpag_df.columns]

    if len(colunas_existentes) < len(colunas_necessarias):
        st.warning("Não foi possível gerar o gráfico de evolução. As seguintes colunas estão ausentes: "
                   + ", ".join(set(colunas_necessarias) - set(colunas_existentes)))
    else:
        evol_df = interpag_df[colunas_necessarias].copy()
        evol_df['Terceirizada / Inter Pag'] = evol_df['Terceirizada / Inter Pag'].str.split('|')
        evol_df = evol_df.explode('Terceirizada / Inter Pag')
        evol_df['Terceirizada / Inter Pag'] = evol_df['Terceirizada / Inter Pag'].str.strip()

        evol_df = evol_df[evol_df['Terceirizada / Inter Pag'].str.lower() == 'inter pag']

        ativos = evol_df[['Data da Distribuição', 'Terceirizada / Inter Pag']].copy()
        ativos['Data'] = pd.to_datetime(ativos['Data da Distribuição'], errors='coerce')
        ativos['Contagem'] = 1

        encerrados = evol_df[
            (evol_df['Transito em Julgado'].str.lower() == 'sim') &
            evol_df['Data da Última Movimentação'].notna()
            ]
        encerrados['Data'] = pd.to_datetime(encerrados['Data da Última Movimentação'], errors='coerce')
        encerrados = encerrados[['Data', 'Terceirizada / Inter Pag']]
        encerrados['Contagem'] = -1

        linha_df = pd.concat([ativos[['Data', 'Terceirizada / Inter Pag', 'Contagem']], encerrados], ignore_index=True)
        linha_df = linha_df.dropna(subset=['Data'])
        linha_df = linha_df.sort_values('Data')
        linha_df = linha_df.groupby(['Data', 'Terceirizada / Inter Pag']).sum().reset_index()
        linha_df['Ativos'] = linha_df.groupby('Terceirizada / Inter Pag')['Contagem'].cumsum()

        linha_df = linha_df.rename(columns={'Terceirizada / Inter Pag': 'Terceirizada'})

        fig_linha = px.line(
            linha_df,
            x='Data',
            y='Ativos',
            color='Terceirizada',
            hover_name='Terceirizada',
            labels={'Ativos': 'Processos Ativos', 'Data': 'Data'},
            title='Evolução de Processos Ativos - Inter Pag'
        )

        st.plotly_chart(fig_linha, use_container_width=True)

# --- Aba Análise de Acordos ---
with tab5:
    st.markdown("# Análise de Acordos Firmados")

    # Filtro: apenas processos com acordo "sim"
    acordos_df = df_full[
        (df_full['Acordo'].str.lower().str.strip() == 'sim') &
        (df_full['Empregado Próprio'].str.lower().str.strip() == 'sim')
        ].copy()

    if acordos_df.empty:
        st.warning("Nenhum acordo encontrado nos dados.")
    else:
        valor_col = 'Pagamento de Condenação/Acordo'
        econ_col = 'Economia Concreta'
        liqui_col = 'Liquidação Inicial Sem Dedução'

        # --- KPIs de resumo geral ---
        total_liquidacao = acordos_df[liqui_col].sum()
        total_pago = acordos_df[liqui_col].sum() - acordos_df[econ_col].sum()
        total_economia = acordos_df[econ_col].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Total da Liquidação Inicial",
                  f"R$ {total_liquidacao:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
        k2.metric("Total Efetivamente Pago",
                  f"R$ {total_pago:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
        k3.metric("Economia Total com Acordos",
                  f"R$ {total_economia:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

        # --- Gráfico comparativo dos totais ---
        st.markdown("### Comparativo Geral dos Acordos")

        df_totais = pd.DataFrame({
            "Categoria": ["Liquidação Inicial", "Valor Pago", "Economia"],
            "Valor": [total_liquidacao, total_pago, total_economia]
        })

        fig_totais = px.bar(
            df_totais,
            x='Categoria',
            y='Valor',
            text='Valor',
            color='Categoria',
            color_discrete_sequence=["#FF6600", "#FF6600", "#FF6600"],
            labels={'Valor': 'R$'}
        )

        fig_totais.update_traces(texttemplate='R$ %{y:,.2f}', textposition='outside')
        fig_totais.update_layout(
            showlegend=False,
            yaxis_title='Valor (R$)',
            xaxis_title='',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_totais, use_container_width=True)

    # Gráficos lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Maiores Acordos (Valor Absoluto)")
        top_valor = acordos_df.nlargest(5, valor_col)
        fig_valor = px.bar(
            top_valor,
            x=valor_col,
            y='Adverso principal',
            orientation='h',
            labels={valor_col: 'Valor do Acordo', 'Adverso principal': 'Reclamante'},
            color_discrete_sequence=["#FF6600"]
        )
        fig_valor.update_layout(xaxis_tickprefix="R$ ", xaxis_tickformat=",")
        fig_valor.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_valor, use_container_width=True)

    with col2:
        st.markdown("### Maiores Acordos (Proveito Econômico)")
        top_econ = acordos_df.nlargest(5, econ_col)
        fig_econ = px.bar(
            top_econ,
            x=econ_col,
            y='Adverso principal',
            orientation='h',
            labels={econ_col: 'Proveito Econômico', 'Adverso principal': 'Reclamante'},
            color_discrete_sequence=["#FF6600"]
        )
        fig_econ.update_layout(xaxis_tickprefix="R$ ", xaxis_tickformat=",")
        fig_econ.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_econ, use_container_width=True)

    # Dropdown ordenado por maior economia
    st.markdown("### Detalhamento de Acordos")
    acordos_df['label'] = acordos_df['Adverso principal'] + ' - R$ ' + acordos_df[valor_col].fillna(0).map(
        '{:,.2f}'.format)
    acordos_df_sorted = acordos_df.sort_values(by=econ_col, ascending=False)
    selecao = st.selectbox("Selecione um Reclamante (Valor do Acordo)", acordos_df_sorted['label'], index=0)

    if selecao:
        linha = acordos_df[acordos_df['label'] == selecao]
        cols_mostrar = [
            'Nº processo principal',
            'Adverso principal',
            'Comarca',
            'Empregado Próprio',
            valor_col,
            liqui_col,
            econ_col
        ]
        linha_mostrar = linha[cols_mostrar].copy()
        linha_mostrar.columns = [
            'Nº Processo',
            'Reclamante',
            'Comarca',
            'Empregado Próprio',
            'Valor do Acordo',
            'Liquidação Inicial',
            'Proveito Econômico'
        ]

        # Formatação monetária
        for col in ['Valor do Acordo', 'Liquidação Inicial', 'Proveito Econômico']:
            linha_mostrar[col] = linha_mostrar[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

        linha_mostrar.index = ['']
        st.dataframe(linha_mostrar, use_container_width=True)

# --- Aba Cláusula 11ª ---
with tab6:
    st.markdown("# Análise da Cláusula 11ª – Convenção Coletiva")

    col_sem = 'Liquidação Inicial Sem Dedução'
    col_com = 'Liquidação Inicial Com Dedução'

    # Filtra apenas os processos ainda sem trânsito em julgado
    df_em_andamento = df_full[df_full['Transito em Julgado'].str.strip().str.lower() != 'sim'].copy()

    # Cria coluna de dedução apenas quando há valor válido com dedução
    df_em_andamento['Deduções CCT'] = df_em_andamento.apply(
        lambda row: row[col_sem] - row[col_com] if row[col_com] > 0 else 0,
        axis=1
    )

    df_validos = df_em_andamento.dropna(subset=['Deduções CCT'])
    df_com_deducao = df_validos[df_validos[col_com] > 0]  # usa apenas processos com cálculo efetivo

    # KPI – Valor total deduzido e percentual médio real
    total_bruto = df_com_deducao[col_sem].sum()
    total_deduzido = df_com_deducao['Deduções CCT'].sum()
    percentual_medio = (total_deduzido / total_bruto) * 100 if total_bruto > 0 else 0

    k1, k2 = st.columns(2)
    k1.metric("Total Reduzido pela Cláusula 11ª",
              f"R$ {total_deduzido:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    k2.metric("Redução Média Percentual", f"{percentual_medio:.1f}%")

    # Gráfico – Top 10 maiores deduções
    st.markdown("## Top 10 Processos com Maior Dedução")
    top_deduc = df_com_deducao[['Adverso principal', 'Deduções CCT']].nlargest(10, 'Deduções CCT')
    fig_top_deduc = px.bar(
        top_deduc,
        x='Deduções CCT',
        y='Adverso principal',
        orientation='h',
        labels={'Deduções CCT': 'Valor Deduzido', 'Adverso principal': 'Reclamante'},
        color_discrete_sequence=["#FF6600"]
    )
    fig_top_deduc.update_layout(xaxis_tickprefix="R$ ", xaxis_tickformat=",")
    fig_top_deduc.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_top_deduc, use_container_width=True)

    # Dropdown com processos ordenados por maior economia
    st.markdown("### Detalhamento por Processo")
    df_com_deducao['label'] = df_com_deducao['Adverso principal'] + ' - R$ ' + df_com_deducao['Deduções CCT'].map(
        '{:,.2f}'.format).str.replace(",", "v").str.replace(".", ",").str.replace("v", ".")

    df_sorted = df_com_deducao.sort_values(by='Deduções CCT', ascending=False)
    selecao = st.selectbox("Selecione um Reclamante (Redução pela Cláusula 11ª)", df_sorted['label'], index=0)

    if selecao:
        linha = df_com_deducao[df_com_deducao['label'] == selecao]
        linha = linha[[
            'Nº processo principal',
            'Adverso principal',
            'Comarca',
            col_sem,
            col_com,
            'Deduções CCT'
        ]].copy()
        linha.columns = [
            'Nº Processo',
            'Reclamante',
            'Comarca',
            'Liquidação Sem Dedução',
            'Liquidação Com Dedução',
            'Redução pela Cláusula 11ª'
        ]

        for col in ['Liquidação Sem Dedução', 'Liquidação Com Dedução', 'Redução pela Cláusula 11ª']:
            linha[col] = linha[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

        linha.index = ['']
        st.dataframe(linha, use_container_width=True)

# --- Aba Êxito (Improcedência) ---
with tab7:
    st.markdown("# Análise de Casos Improcedentes e Economia Processual")

    # Prepara base
    df = df_full.copy()

    # Padroniza colunas
    df['Procedência Atual'] = df['Procedência Atual'].astype(str).str.strip().str.lower()
    df['Transito em Julgado'] = df['Transito em Julgado'].astype(str).str.strip().str.lower()
    df['Acordo'] = df['Acordo'].astype(str).str.strip().str.lower()
    df['Economia Concreta'] = pd.to_numeric(df['Economia Concreta'], errors='coerce').fillna(0)
    df['Economia Potencial'] = pd.to_numeric(df['Economia Potencial'], errors='coerce').fillna(0)

    # --- Bloco 1: Improcedentes ---
    improcedentes_df = df[df['Procedência Atual'] == 'improcedente'].copy()

    total_improcedentes = len(improcedentes_df)
    nao_transitados_df = improcedentes_df[improcedentes_df['Transito em Julgado'] == 'não']
    transitados_df = improcedentes_df[
        (improcedentes_df['Transito em Julgado'] == 'sim') &
        (improcedentes_df['Acordo'] != 'sim')
    ]

    soma_economia_concreta = transitados_df['Economia Concreta'].sum()

    # --- Bloco 2: Economia Potencial Global (processos não transitados, sem acordo) ---
    nao_transitados_geral = df[
        (df['Transito em Julgado'] == 'não') &
        (df['Acordo'] != 'sim')
    ]
    soma_economia_potencial = nao_transitados_geral['Economia Potencial'].sum()

    # KPIs
    st.markdown("### Resumo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Improcedentes", total_improcedentes)
    col2.metric("Improcedentes Não Julgados", len(nao_transitados_df))
    col3.metric("Improcedentes Julgados", total_improcedentes - len(nao_transitados_df))

    col4, col5 = st.columns(2)
    col4.metric("Economia Concreta (Improcedentes Julgados)",
                f"R$ {soma_economia_concreta:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    col5.metric("Economia Potencial (Todos Não Julgados)",
                f"R$ {soma_economia_potencial:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

    # Detalhamento por processo improcedente
    st.markdown("### Detalhamento dos Casos Improcedentes")
    improcedentes_df['label'] = improcedentes_df['Adverso principal'] + ' - ' + improcedentes_df['Comarca']
    selecao = st.selectbox("Selecione um Reclamante", improcedentes_df['label'].sort_values(), index=0)

    if selecao:
        linha = improcedentes_df[improcedentes_df['label'] == selecao]
        tabela = linha[[
            'Nº processo principal',
            'Adverso principal',
            'Comarca',
            'Procedência Atual',
            'Transito em Julgado',
            'Acordo',
            'Economia Potencial',
            'Economia Concreta'
        ]].copy()

        tabela.columns = [
            'Nº Processo',
            'Reclamante',
            'Comarca',
            'Decisão',
            'Transitado em Julgado',
            'Acordo?',
            'Economia Potencial',
            'Economia Concreta'
        ]

        for col in ['Economia Potencial', 'Economia Concreta']:
            tabela[col] = tabela[col].apply(lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

        tabela.index = ['']
        st.dataframe(tabela, use_container_width=True)

        # Gráfico de pizza: distribuição da procedência dos casos ainda não julgados
        st.markdown("### Distribuição da Procedência (Casos Não Julgados)")

        nao_julgados = df[df['Transito em Julgado'] == 'não'].copy()
        nao_julgados['Procedência Atual'] = nao_julgados['Procedência Atual'].str.title()

        # Define categorias principais e consolida o restante em "Outros"
        principais = ['Improcedente', 'Parcialmente Procedente', 'Procedente', 'Fase De Conhecimento']
        nao_julgados['Categoria'] = nao_julgados['Procedência Atual'].apply(
            lambda x: x if x in principais else 'Outros'
        )

        # Agrupa os dados para o gráfico
        dist_proc = nao_julgados['Categoria'].value_counts().reset_index()
        dist_proc.columns = ['Procedência', 'Quantidade']

        # Gera o gráfico de pizza
        fig_pizza = px.pie(
            dist_proc,
            values='Quantidade',
            names='Procedência',
            title='Distribuição da Procedência entre os Processos Sem Trânsito em Julgado',
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        fig_pizza.update_traces(textinfo='label+percent', pull=[0.05] * len(dist_proc))

        fig_pizza.update_layout(width=800, height=600)

        st.plotly_chart(fig_pizza, use_container_width=True)
