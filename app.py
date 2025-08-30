#Desafio Ciencia de dados.
import streamlit as st
import pandas as pd
import plotly.express as px

# Aqui está a configuração da página
st.set_page_config(
    page_title=" PProductions",
    page_icon="🎬",
    layout="wide",
)

# Arquivo fornecido para realizar o carregamento dos dados.
df = pd.read_csv("desafio_indicium_imdb.csv",index_col=0)

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

#Aqui está as métricas principais do Dashboard
st.header("🎬 Dashboard - Análise de Filmes")
st.markdown('Explore dados de filmes. Use os filtros à esquerda para refinar sua análise.')
st.markdown('---')


if not df_filtrado.empty:
    nota_media = df_filtrado['IMDB_Rating'].mean()
    bilheteria_total = df_filtrado['Gross'].sum()
    filme_maior_nota = df_filtrado.loc[df_filtrado['IMDB_Rating'].idxmax()]['Series_Title']
    diretor_mais_frequente = df_filtrado['Director'].mode()[0]
    filme_mais_popular = df_filtrado.loc[df_filtrado['No_of_Votes'].idxmax()]['Series_Title']
else:
    nota_media, bilheteria_total, filme_maior_nota, diretor_mais_frequente, filme_mais_popular = 0, 0, 'N/A', 'N/A'

#Layout
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Nota Média', f'{nota_media:.2f}')
col2.metric('Bilheteria Total', f'${bilheteria_total:,.0f}')
col3.metric('Filme Mais Bem Avaliado', filme_maior_nota)
col4.metric('Diretor Mais Frequente', diretor_mais_frequente)
col5.metric('Filme Mais Popular', filme_mais_popular)


st.markdown('---')

#Gráficos para facilitar a visualização:
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


# Series_Title – Nome do filme
# Released_Year - Ano de lançamento
# Certificate - Classificação etária
# Runtime – Tempo de duração
# Genre - Gênero
# IMDB_Rating - Nota do IMDB
# Overview - Overview do filme
# Meta_score - Média ponderada de todas as críticas 
# Director – Diretor
# Star1 - Ator/atriz #1
# Star2 - Ator/atriz #2
# Star3 - Ator/atriz #3
# Star4 - Ator/atriz #4
# No_of_Votes - Número de votos
# Gross - Faturamento


if not df_filtrado.empty:
    st.dataframe(df_filtrado [ [
    'Series_Title', 
    'Released_Year', 
    'Certificate', 
    'Runtime', 
    'Genre', 
    'IMDB_Rating', 
    'Overview', 
    'Meta_score', 
    'Director', 
    'Star1', 
    'Star2', 
    'Star3', 
    'Star4', 
    'No_of_Votes', 
    'Gross'] ] )
else:
    st.warning("Nenhum filme encontrado para os filtros selecionados.")


#Verificar Dataframe:
print(df_filtrado.columns)
