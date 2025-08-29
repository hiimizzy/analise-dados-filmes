#Desafio Ciencia de dados.
import streamlit as st
import pandas as pd
import plotly.express as px

# Aqui está a configuração da página
st.set_page_config(
    page_title="Dashboard IMDb",
    page_icon="🎬",
    layout="wide",
)

# Arquivo fornecido para realizar o carregamento dos dados.
df = pd.read_csv("desafio_indicium_imdb.csv")

# Aqui está a limpeza e transformação de dados
df['Gross'] = df['Gross'].str.replace(',', '').str.replace('$', '').astype(float)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# Barra lateral - filtros
st.sidebar.header("🔍 Filtros")

# Filtro de Ano
anos_disponiveis = sorted(df['Released_Year'].dropna().unique())
anos_selecionados = st.sidebar.multiselect('Ano de Lançamento', anos_disponiveis, default=anos_disponiveis[-5:])

# Filtro de Gênero dos Filmes
generos_disponiveis = sorted(set([genero for sublist in df['Genre'].str.split(', ') for genero in sublist]))
generos_selecionados = st.sidebar.multiselect('Gênero', generos_disponiveis, default=['Drama'])

# Filtro de Diretor
diretores_disponiveis = sorted(df['Director'].unique())
diretores_selecionados = st.sidebar.multiselect('Diretor', diretores_disponiveis)

# Filtro de Certificação
certificacoes = sorted(df['Certificate'].dropna().unique())
certificacoes_selecionadas = st.sidebar.multiselect('Certificação', certificacoes, default=certificacoes)

#Filtragem do DataFrame
df_filtrado = df[
    (df['Released_Year'].isin(anos_selecionados)) &
    (df['Certificate'].isin(certificacoes_selecionadas)) &
    (df['Director'].isin(diretores_selecionados) if diretores_selecionados else True)
]

# Filtro de gênero do filme:
if generos_selecionados:
    df_filtrado = df_filtrado[df_filtrado['Genre'].apply(
        lambda x: any(genero in x for genero in generos_selecionados)
    )]

# --- Métricas Principais ---
st.header("🎬 Dashboard IMDb - Análise de Filmes")
st.markdown('Explore dados de filmes do IMDb. Utilize os filtros à esquerda para refinar sua análise.')

if not df_filtrado.empty:
    nota_media = df_filtrado['IMDB_Rating'].mean()
    bilheteria_total = df_filtrado['Gross'].sum()
    filme_maior_nota = df_filtrado.loc[df_filtrado['IMDB_Rating'].idxmax()]['Series_Title']
    diretor_mais_frequente = df_filtrado['Director'].mode()[0]
else:
    nota_media, bilheteria_total, filme_maior_nota, diretor_mais_frequente = 0, 0, 'N/A', 'N/A'

col1, col2, col3, col4 = st.columns(4)
col1.metric('Nota Média', f'{nota_media:.2f}')
col2.metric('Bilheteria Total', f'${bilheteria_total:,.0f}')
col3.metric('Filme Mais Bem Avaliado', filme_maior_nota)
col4.metric('Diretor Mais Frequente', diretor_mais_frequente)

st.markdown('---')

#Gráficos para facilitar a visualização
st.subheader('Visualizações')
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    if not df_filtrado.empty:
        top_filmes = df_filtrado.nlargest(10, 'IMDB_Rating')[['Series_Title', 'IMDB_Rating']]
        fig = px.bar(top_filmes, 
                     x='IMDB_Rating', 
                     y='Series_Title', 
                     orientation='h',
                     title='Top 10 Filmes por Avaliação',
                     labels={'IMDB_Rating': 'Nota IMDb', 'Series_Title': 'Filme'})
        st.plotly_chart(fig, use_container_width=True)

with col_graf2:
    if not df_filtrado.empty:
        fig = px.scatter(df_filtrado,
                         x='Runtime',
                         y='IMDB_Rating',
                         title='Relação entre Duração e Avaliação',
                         labels={'Runtime': 'Duração (minutos)', 'IMDB_Rating': 'Nota IMDb'})
        st.plotly_chart(fig, use_container_width=True)

col_graf3, col_graf4 = st.columns(2)

with col_graf3:
    if not df_filtrado.empty:
        fig = px.histogram(df_filtrado,
                           x='IMDB_Rating',
                           nbins=20,
                           title='Distribuição das Avaliações',
                           labels={'IMDB_Rating': 'Nota IMDb'})
        st.plotly_chart(fig, use_container_width=True)

with col_graf4:
    if not df_filtrado.empty:
        avaliacao_por_ano = df_filtrado.groupby('Released_Year')['IMDB_Rating'].mean().reset_index()
        fig = px.line(avaliacao_por_ano,
                      x='Released_Year',
                      y='IMDB_Rating',
                      title='Evolução da Avaliação Média por Ano',
                      labels={'Released_Year': 'Ano', 'IMDB_Rating': 'Nota Média'})
        st.plotly_chart(fig, use_container_width=True)

#Tabela de Dados 
st.subheader("Tabela de Filmes")
st.dataframe(df_filtrado[['Series_Title', 'Released_Year', 'Genre', 'Director', 'IMDB_Rating', 'Gross']])