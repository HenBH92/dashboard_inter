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
    for comarca in comarcas.dropna().unique():
        try:
            loc = geolocator.geocode(f"{comarca}, Brasil")
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

df_full['Deduções CCT'] = df_full['Liquidação Inicial Sem Dedução'] - df_full['Liquidação Inicial Com Dedução']

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    "Análise de Objetos",
    "Análise de Terceirizadas",
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
    k1, k2, k3 = st.columns(3)
    k1.metric("Total de Processos", f"{total_processos}")
    k2.metric("Comarcas com Processos", f"{num_comarcas}")
    k3.metric("Maior Concentração", f"{comarca_max} ({max_processos})")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Proveito Econômico (R$)", f"{econ_total:,.2f}")
    k6.metric("Total Garantido (R$)", f"{dep_total:,.2f}")
    k7.metric("Acordos Firmados", f"{num_acordos}")
    economia_cct = df_full['Deduções CCT'].dropna().sum()
    k8.metric("Economia CCT (R$)", f"{economia_cct:,.2f}")

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Distribuição Geográfica</h3>", unsafe_allow_html=True)

    col_map, col_ranking = st.columns([3, 2])
    with col_map:
        show_heatmap = st.toggle("Exibir Heatmap", key="tab1_heatmap")

        view = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5, pitch=0)
        layers = [
            pdk.Layer("GeoJsonLayer", data="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson", stroked=True, filled=False, get_line_color=[100, 100, 100], line_width_min_pixels=1),
            pdk.Layer("ScatterplotLayer", data=map_df, get_position=["lon", "lat"], get_fill_color=[255, 102, 0, 220], get_radius=20000, pickable=True, auto_highlight=True),
            pdk.Layer("TextLayer", data=map_df, get_position=["lon", "lat"], get_text="Processos", get_color=[50, 50, 50], get_size=12, get_angle=0, get_text_anchor="middle")
        ]
        if show_heatmap:
            layers.append(pdk.Layer("HeatmapLayer", data=map_df, get_position=["lon", "lat"], aggregation="SUM"))

        deck = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", initial_view_state=view, layers=layers, tooltip={"text": "Comarca: {Comarca}\nProcessos: {Processos}"})
        st.pydeck_chart(deck, use_container_width=True)

    with col_ranking:
        st.markdown("<h3 style='color:#201747'>Top 10 Comarcas</h3>", unsafe_allow_html=True)
        top10 = counts.nlargest(10, 'Processos')
        fig1 = px.bar(top10, x='Processos', y='Comarca', orientation='h', color_discrete_sequence=["#FF6600"])
        fig1.update_yaxes(categoryorder='array', categoryarray=top10['Comarca'][::-1].tolist())
        fig1.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Processos por Comarca</h3>", unsafe_allow_html=True)
    comarcas_list = counts.sort_values(by='Processos', ascending=False)['Comarca']
    selected = st.selectbox("Selecione uma Comarca", comarcas_list)
    df_sel = df_full[df_full['Comarca'] == selected][['Nº processo principal', 'Adverso principal', 'Empregado Próprio']].copy()
    df_sel.index = range(1, len(df_sel) + 1)
    st.dataframe(df_sel, use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Processos Distribuídos por Mês</h3>", unsafe_allow_html=True)
    ts = df_full.dropna(subset=[dist_col]).groupby(df_full[dist_col].dt.to_period('M')).size().reset_index(name='Count')
    ts[dist_col] = ts[dist_col].dt.to_timestamp()
    ts['Month'] = ts[dist_col].dt.strftime('%Y-%m')
    fig2 = px.bar(ts, x='Month', y='Count', color_discrete_sequence=["#FF6600"], labels={'Count': 'Processos'})
    fig2.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Evolução da Carteira Ativa</h3>", unsafe_allow_html=True)
    df_dist = df_full.dropna(subset=[dist_col]).copy()
    df_dist['Mês'] = df_dist[dist_col].dt.to_period('M').dt.to_timestamp()
    df_dist['Delta'] = 1
    df_fech = df_full[df_full['Status'] == 'Inativa'].dropna(subset=[last_mov_col]).copy()
    df_fech['Mês'] = df_fech[last_mov_col].dt.to_period('M').dt.to_timestamp()
    df_fech['Delta'] = -1
    df_evo = pd.concat([df_dist[['Mês', 'Delta']], df_fech[['Mês', 'Delta']]])
    df_evo = df_evo.groupby('Mês').sum().sort_index().cumsum().reset_index()
    df_evo.columns = ['Mês', 'Processos Ativos']
    fig_carteira = px.line(df_evo, x='Mês', y='Processos Ativos', markers=True, title="Evolução da Carteira (Processos Ativos)", labels={'Mês': 'Mês', 'Processos Ativos': 'Total Acumulado'}, color_discrete_sequence=["#FF6600"])
    fig_carteira.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    st.plotly_chart(fig_carteira, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#201747'>Evolução do Risco</h3>", unsafe_allow_html=True)
    open_ev = df_full.dropna(subset=[dist_col])[[dist_col, dlq_col]].copy()
    open_ev['month'] = open_ev[dist_col].dt.to_period('M').dt.to_timestamp()
    open_ev['delta'] = open_ev[dlq_col]
    closed_ev = df_full[df_full['Status'] == 'Inativa'].dropna(subset=[last_mov_col])[[last_mov_col, dlq_col]].copy()
    closed_ev['month'] = closed_ev[last_mov_col].dt.to_period('M').dt.to_timestamp()
    closed_ev['delta'] = -closed_ev[dlq_col]
    events = pd.concat([open_ev[['month', 'delta']], closed_ev[['month', 'delta']]], ignore_index=True)
    monthly = events.groupby('month')['delta'].sum().sort_index()
    risk = monthly.cumsum().reset_index(name='Risk')
    risk['MA3'] = risk['Risk'].rolling(3, min_periods=1).mean()
    fig_risk = px.bar(risk, x='month', y='Risk', color_discrete_sequence=["#FF6600"], labels={'Risk': 'Risco Exposto', 'month': 'Mês'}, title="Evolução do Risco Mensal")
    fig_risk.add_scatter(x=risk['month'], y=risk['MA3'], mode='lines', name='Média Móvel 3M')
    fig_risk.update_layout(title_x=0.5, plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF')
    st.plotly_chart(fig_risk, use_container_width=True)

# --- Aba Análise de Objetos ---
with tab2:
    st.markdown("<h2 style='color:#201747'>Análise de Objetos dos Pedidos</h2>", unsafe_allow_html=True)

    obj_cols = [c for c in df_full.columns if 'objeto' in c.lower()]
    if obj_cols:
        obj_col = obj_cols[0]
        s = df_full[obj_col].dropna().str.split('|').explode().str.strip()
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
            title="Top 10 Objetos com Maior Incidência",
            title_font=dict(size=18, color="#201747"),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_obj, use_container_width=True)

        st.markdown("<h4 style='color:#201747'>Filtrar por Objeto</h4>", unsafe_allow_html=True)
        if st.checkbox("Selecionar todos", value=False):
            sel_objs = obj_counts['Objeto'].tolist()
        else:
            sel_objs = st.multiselect("Selecionar objetos específicos", obj_counts['Objeto'].tolist(),
                                      default=obj_counts['Objeto'].head(3).tolist())

        if sel_objs:
            mask = pd.Series(True, index=df_full.index)
            for o in sel_objs:
                mask &= df_full[obj_col].str.contains(o, na=False)
            df_obj = df_full[mask]
            cols_show = ['Nº processo principal', 'Adverso principal', 'Empregado Próprio', obj_col]
            st.markdown(f"<h4 style='color:#201747'>Processos com objetos: {', '.join(sel_objs)}</h4>", unsafe_allow_html=True)
            st.dataframe(df_obj[cols_show].reset_index(drop=True), use_container_width=True)

    else:
        st.info("Coluna de objetos não encontrada")

    st.markdown("---")
    st.markdown("<h2 style='color:#201747'>Radar de Objetos em Casos Procedentes e Parcialmente Procedentes</h2>", unsafe_allow_html=True)

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
        st.caption("Os objetos representados neste gráfico foram os mais frequentes em sentenças procedentes ou parcialmente procedentes.")
    else:
        st.info("Coluna de objeto não encontrada para gerar o radar.")


# --- Aba Análise de Terceirizadas ---
with tab3:
    st.markdown("# Análise de Processos (Terceiros ou Inter Pag)")

    # Filtra somente não empregados próprios
    terceiro_df = df_full[df_full['Empregado Próprio'] != 'Sim']
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
    cols_show = ['Nº processo principal', 'Comarca', 'Adverso principal', 'Terceirizada / Inter Pag']
    st.dataframe(terceiro_df[cols_show].reset_index(drop=True), use_container_width=True, height=300)

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

# --- Aba Análise de Acordos ---
with tab4:
    st.markdown("# Análise de Acordos Firmados")

    # Filtro: apenas processos com acordo "sim"
    acordos_df = df_full[df_full['Acordo'].str.lower().str.strip() == 'sim'].copy()

    if acordos_df.empty:
        st.warning("Nenhum acordo encontrado nos dados.")
    else:
        valor_col = 'Pagamento de Condenação/Acordo'
        econ_col = 'Economia Concreta'
        liqui_col = 'Liquidação Inicial Sem Dedução'

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

    # Proporção de Acordos por Tipo de Empregado
    st.markdown("### Perfil Empregatício dos Acordos")

    # Converte para rótulos amigáveis
    acordos_df['Tipo de Empregado'] = acordos_df['Empregado Próprio'].apply(
        lambda x: 'Empregado Próprio' if str(x).strip().lower() == 'sim' else 'Terceirizado'
    )

    perfil_acordos = acordos_df['Tipo de Empregado'].value_counts().reset_index()
    perfil_acordos.columns = ['Tipo de Empregado', 'Quantidade']

    fig_emp = px.pie(
        perfil_acordos,
        names='Tipo de Empregado',
        values='Quantidade',
        title='Distribuição dos Acordos por Tipo de Empregado',
        color_discrete_sequence=["#FF6600", "#999999"]
    )
    st.plotly_chart(fig_emp, use_container_width=True)

# --- Aba Cláusula 11ª ---
# --- Aba Cláusula 11ª ---
with tab5:
    st.markdown("# Análise da Cláusula 11ª – Convenção Coletiva")

    col_sem = 'Liquidação Inicial Sem Dedução'
    col_com = 'Liquidação Inicial Com Dedução'

    # Cria coluna de diferença
    df_full['Deduções CCT'] = df_full[col_sem] - df_full[col_com]
    df_validos = df_full.dropna(subset=['Deduções CCT'])

    # KPI – Valor total deduzido e percentual médio
    total_deduzido = df_validos['Deduções CCT'].sum()
    percentual_medio = (df_validos['Deduções CCT'] / df_validos[col_sem]).mean() * 100

    k1, k2 = st.columns(2)
    k1.metric("Total Reduzido pela Cláusula 11ª",
              f"R$ {total_deduzido:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    k2.metric("Redução Média Percentual", f"{percentual_medio:.1f}%")

    # Gráfico – Top 10 maiores deduções
    st.markdown("## Top 10 Processos com Maior Dedução")
    top_deduc = df_validos[['Adverso principal', 'Deduções CCT']].nlargest(10, 'Deduções CCT')
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
    df_validos['label'] = df_validos['Adverso principal'] + ' - R$ ' + df_validos['Deduções CCT'].fillna(0).map(
        '{:,.2f}'.format)
    df_sorted = df_validos.sort_values(by='Deduções CCT', ascending=False)
    selecao = st.selectbox("Selecione um Reclamante (Redução pela Cláusula 11ª)", df_sorted['label'], index=0)

    if selecao:
        linha = df_validos[df_validos['label'] == selecao]
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
with tab6:
    st.markdown("# Análise de Êxito – Economia Obtida e Projetada")

    # Prepara base
    exito_df = df_full.copy()
    exito_df['Economia Potencial'] = exito_df['Economia Potencial'].fillna(0)
    exito_df['Economia Concreta'] = exito_df['Economia Concreta'].fillna(0)
    exito_df['Transitado'] = exito_df['Transito em Julgado'].str.strip().str.lower() == 'sim'
    exito_df['Procedência Atual'] = exito_df['Procedência Atual'].astype(str).str.strip().str.title()
    exito_df['Empregado Próprio'] = exito_df['Empregado Próprio'].astype(str).str.strip().str.lower()
    exito_df['Terceirizada / Inter Pag'] = exito_df['Terceirizada / Inter Pag'].fillna('').astype(str).str.lower()


    # Regras de inclusão
    def incluir(row):
        empregado = row['Empregado Próprio'] == 'sim'
        terceirizado = not empregado or 'inter pag' in row['Terceirizada / Inter Pag']
        if empregado:
            return True
        if terceirizado and str(row['Procedência Atual']).strip().lower() == 'improcedente':
            return True
        return False


    exito_df = exito_df[exito_df.apply(incluir, axis=1)]

    exito_df = exito_df[exito_df['Acordo'].str.strip().str.lower() != 'sim']

    # Apenas se houver economia concreta > 0
    exito_df = exito_df[exito_df['Economia Concreta'] > 0]

    # Classificação: consolidada ou potencial
    exito_df['Tipo Economia'] = exito_df['Transitado'].map({
        True: 'Consolidada',
        False: 'Potencial'
    })

    # KPIs
    total_casos = len(exito_df)
    total_consolidada = exito_df[exito_df['Tipo Economia'] == 'Consolidada']['Economia Concreta'].sum()
    total_potencial = exito_df[exito_df['Tipo Economia'] == 'Potencial']['Economia Concreta'].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Casos com Economia", total_casos)
    k2.metric("Economia Consolidada (R$)",
              f"{total_consolidada:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
    k3.metric("Economia Potencial (R$)",
              f"{total_potencial:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))

    # Legenda
    st.markdown("""
    🔸 **Critérios de Análise:**  
    - Utiliza exclusivamente as colunas **Economia Concreta** e **Economia Potencial** (pré-calculadas).  
    - Casos de **terceirizados** (ou **Inter Pag**) **somente entram se forem improcedentes**.  
    - A economia concreta é classificada como **consolidada** se houver trânsito em julgado e **potencial** se ainda não houve.
    """)

    # Gráfico comparativo consolidada vs potencial
    st.markdown("### Comparativo de Economia")
    comp = pd.DataFrame({
        'Categoria': ['Consolidada', 'Potencial'],
        'Valor': [total_consolidada, total_potencial]
    })
    fig_comp = px.bar(
        comp,
        x='Categoria',
        y='Valor',
        labels={'Valor': 'R$'},
        color='Categoria',
        color_discrete_map={
            'Consolidada': '#2E8B57',
            'Potencial': '#999999'
        }
    )
    fig_comp.update_layout(yaxis_tickprefix="R$ ", yaxis_tickformat=",")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Evolução mensal
    st.markdown("### Evolução Mensal da Economia")
    exito_df['Data Mov'] = pd.to_datetime(exito_df['Data da Última Movimentação'], errors='coerce')
    exito_df = exito_df.dropna(subset=['Data Mov'])
    exito_df['Mês'] = exito_df['Data Mov'].dt.to_period('M').dt.to_timestamp()
    mensal = exito_df.groupby(['Mês', 'Tipo Economia'])['Economia Concreta'].sum().reset_index()
    fig_mensal = px.bar(
        mensal,
        x='Mês',
        y='Economia Concreta',
        color='Tipo Economia',
        barmode='group',
        labels={'Economia Concreta': 'R$', 'Mês': 'Mês'},
        color_discrete_map={
            'Consolidada': '#2E8B57',
            'Potencial': '#999999'
        }
    )
    fig_mensal.update_layout(yaxis_tickprefix="R$ ", yaxis_tickformat=",")
    st.plotly_chart(fig_mensal, use_container_width=True)

    # Detalhamento
    st.markdown("### Detalhamento por Processo")
    exito_df['label'] = exito_df['Adverso principal'] + ' - R$ ' + exito_df['Economia Concreta'].apply(
        lambda x: f"{x:,.2f}")
    selected = st.selectbox("Selecione um Reclamante",
                            exito_df.sort_values(by='Economia Concreta', ascending=False)['label'])

    if selected:
        linha = exito_df[exito_df['label'] == selected]
        tabela = linha[[
            'Nº processo principal',
            'Adverso principal',
            'Comarca',
            'Procedência Atual',
            'Empregado Próprio',
            'Terceirizada / Inter Pag',
            'Economia Potencial',
            'Economia Concreta',
            'Tipo Economia'
        ]].copy()
        tabela.columns = [
            'Nº Processo',
            'Reclamante',
            'Comarca',
            'Decisão',
            'Empregado Próprio',
            'Terceirizada / Inter Pag',
            'Economia Potencial',
            'Economia Concreta',
            'Situação da Economia'
        ]
        for col in ['Economia Potencial', 'Economia Concreta']:
            tabela[col] = tabela[col].apply(
                lambda x: f"R$ {x:,.2f}".replace(",", "v").replace(".", ",").replace("v", "."))
        tabela.index = ['']
        st.dataframe(tabela, use_container_width=True)
