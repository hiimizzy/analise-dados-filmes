# Desafio Ciência de Dados - PProductions
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Aqui está a configuração da página
st.set_page_config(
    page_title="PProductions - Analytics",
    page_icon="🎬",
    layout="wide",
)

# Arquivo fornecido para realizar o carregamento dos dados.
df = pd.read_csv("desafio_indicium_imdb.csv", index_col=0)

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

# Filtragem do DataFrame
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

# Aqui estão as métricas principais do Dashboard:
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
    nota_media, bilheteria_total, filme_maior_nota, diretor_mais_frequente, filme_mais_popular = 0, 0, 'N/A', 'N/A', 'N/A'

# Layout 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Nota Média', f'{nota_media:.2f}')
col2.metric('Bilheteria Total', f'${bilheteria_total:,.0f}')
col3.metric('Filme Mais Bem Avaliado', filme_maior_nota)
col4.metric('Diretor Mais Frequente', diretor_mais_frequente)
col5.metric('Filme Mais Popular', filme_mais_popular)

st.markdown('---')

# Gráficos para facilitar a visualização:
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
                     labels={'IMDB_Rating': 'Nota IMDb', 'Series_Title': 'Filme'},
                     color_discrete_sequence=["orange"])
        st.plotly_chart(fig, use_container_width=True)

with col_graf2:
    if not df_filtrado.empty:
        fig = px.scatter(df_filtrado,
                         x='Runtime',
                         y='IMDB_Rating',
                         title='Relação entre Duração e Avaliação',
                         labels={'Runtime': 'Duração (minutes)', 'IMDB_Rating': 'Nota IMDb'},
                         color_discrete_sequence=["orange"])
        st.plotly_chart(fig, use_container_width=True)

col_graf3, col_graf4 = st.columns(2)

with col_graf3:
    if not df_filtrado.empty:
        fig = px.histogram(df_filtrado,
                           x='IMDB_Rating',
                           nbins=20,
                           title='Distribuição das Avaliações',
                           labels={'IMDB_Rating': 'Nota IMDb'},
                           color_discrete_sequence=["orange"])
        st.plotly_chart(fig, use_container_width=True)

with col_graf4:
    if not df_filtrado.empty:
        avaliacao_por_ano = df_filtrado.groupby('Released_Year')['IMDB_Rating'].mean().reset_index()
        fig = px.line(avaliacao_por_ano,
                      x='Released_Year',
                      y='IMDB_Rating',
                      title='Evolução da Avaliação Média por Ano',
                      labels={'Released_Year': 'Ano', 'IMDB_Rating': 'Nota Média'},
                      color_discrete_sequence=["orange"])
        st.plotly_chart(fig, use_container_width=True)

# Análise de Texto (NLP) e com grafico:
st.markdown('---')
st.subheader('📊 Análise de Texto - Overview dos Filmes')

if not df_filtrado.empty:
    try:
        from wordcloud import WordCloud
        
        df_filtrado['Overview'] = df_filtrado['Overview'].fillna('')

        st.write("**Nuvem de Palavras das Overviews**")
        all_overviews = " ".join(overview for overview in df_filtrado['Overview'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_overviews)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        st.write("**Análise de Palavras por Gênero**")
        genre_option = st.selectbox('Selecione um gênero para análise:', generos_disponiveis)
        
        if genre_option:
            genre_movies = df_filtrado[df_filtrado['Genre'].str.contains(genre_option, na=False)]
            if not genre_movies.empty:
                genre_overviews = " ".join(overview for overview in genre_movies['Overview'])
                genre_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genre_overviews)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(genre_wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Palavras Mais Frequentes em Filmes de {genre_option}')
                st.pyplot(fig)
                
                # Top palavras
                from collections import Counter
                import re
                
                words = re.findall(r'\w+', genre_overviews.lower())
                common_words = Counter(words).most_common(10)
                
                common_df = pd.DataFrame(common_words, columns=['Palavra', 'Frequência'])
                fig = px.bar(common_df, x='Palavra', y='Frequência', 
                             title=f'Top 10 Palavras em Filmes de {genre_option}',
                             color_discrete_sequence=["orange"])
                st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Biblioteca WordCloud não instalada. Execute: pip install wordcloud")

# Modelo Preditivo e Previsão
st.markdown('---')
st.subheader('🎯 Modelo Preditivo de Notas IMDB')

if st.button('Treinar Modelo Preditivo'):
    with st.spinner('Treinando modelo...'):
        # Preparo dos dados para modelagem
        df_model = df.copy()

        categorical_cols = ['Certificate', 'Genre', 'Director'] + [f'Star{i}' for i in range(1, 5)]
        for col in categorical_cols:
            df_model[col] = df_model[col].astype('category').cat.codes

        features = ['Runtime', 'Meta_score', 'No_of_Votes', 'Gross'] + categorical_cols
        X = df_model[features].fillna(0)
        y = df_model['IMDB_Rating']

        st.session_state['X_columns'] = X.columns.tolist()
        st.session_state['df_model'] = df_model
        
        # Treinar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Salvar modelo
        joblib.dump(model, 'imdb_rating_predictor.pkl')
        
        st.success(f'🎉 Modelo treinado com sucesso! MAE: {mae:.3f}, R²: {r2:.3f}')
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), x='importance', y='feature', 
                     title='Top 10 Features Mais Importantes')
        st.plotly_chart(fig, use_container_width=True)

# Previsão para o filme The Shawshank Redemption:
st.markdown('---')
st.subheader('🔮 Previsão para "The Shawshank Redemption"')

try:
    model = joblib.load('imdb_rating_predictor.pkl')
    
    # Dados do filme 'The Shawshank Redemption' fornecido no documento:
    shawshank_data = {
        'Series_Title': 'The Shawshank Redemption',
        'Released_Year': 1994,
        'Certificate': 'A',
        'Runtime': 142,
        'Genre': 'Drama',
        'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'Meta_score': 80.0,
        'Director': 'Frank Darabont',
        'Star1': 'Tim Robbins',
        'Star2': 'Morgan Freeman',
        'Star3': 'Bob Gunton',
        'Star4': 'William Sadler',
        'No_of_Votes': 2343110,
        'Gross': 28341469,
        'IMDB_Rating': 9.3 
    }
    
    # Criar um DataFrame temporário para o filme:
    df_temp = pd.concat([df, pd.DataFrame([shawshank_data])], ignore_index=True)

    df_model_temp = df_temp.copy()
    categorical_cols = ['Certificate', 'Genre', 'Director'] + [f'Star{i}' for i in range(1, 5)]
    for col in categorical_cols:
        df_model_temp[col] = df_model_temp[col].astype('category').cat.codes

    # Separar a linha do filme
    shawshank_row = df_model_temp[df_model_temp['Series_Title'] == 'The Shawshank Redemption'].iloc[0]
    
    # Selecionar as features
    features = ['Runtime', 'Meta_score', 'No_of_Votes', 'Gross'] + categorical_cols
    X_columns = st.session_state.get('X_columns', features)  # Se não tiver, usa a lista features
    
    # Garantir a mesma ordem
    shawshank_features = shawshank_row[X_columns].fillna(0).values.reshape(1, -1)
    
    # Fazer a previsão do filme:
    predicted_rating = model.predict(shawshank_features)[0]
    actual_rating = shawshank_row['IMDB_Rating']
    
    col1, col2 = st.columns(2)
    col1.metric("Nota Real", f"{actual_rating:.1f}")
    col2.metric("Nota Prevista", f"{predicted_rating:.1f}",  
                f"{predicted_rating - actual_rating:.1f}")
    
    st.write(f"**Diferença**: {abs(predicted_rating - actual_rating):.2f} pontos")
    
except FileNotFoundError:
    st.warning("Treine o modelo primeiro para fazer previsões.")
except Exception as e:
    st.error(f"Erro ao fazer previsão: {str(e)}")
    st.info("Certifique-se de treinar o modelo primeiro e que todas as colunas necessárias estão presentes.")

# Respostas às perguntas presente no documento do desafio:
st.markdown('---')
st.subheader('📋 Respostas às Perguntas do Desafio')

with st.expander("1. Qual filme recomendaria para uma pessoa que você não conhece?"): #inclusive ,joker, é uns dos meus filmes favorito.
    st.write(f"""
    Recomendo **"🃏{filme_mais_popular}"** pois o filme possui as seguintes características: 
    - Maior número de votos: {df[df['Series_Title'] == filme_mais_popular]['No_of_Votes'].values[0]:,} votos
    - Nota IMDB: {df[df['Series_Title'] == filme_mais_popular]['IMDB_Rating'].values[0]}/10
    - Gênero: {df[df['Series_Title'] == filme_mais_popular]['Genre'].values[0]}
    - Esta escolha é baseada na análise dos dados fornecidos no desafio.
    """)

with st.expander("2. Quais são os principais fatores relacionados com alta expectativa de faturamento?"):
    # Calcular correlações em tempo real
    numeric_cols = ['Gross', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Runtime']
    correlation_matrix = df[numeric_cols].corr()
    gross_correlations = correlation_matrix['Gross'].sort_values(ascending=False)
    
    st.write("Com base na análise de correlação dos dados:")
    for feature, correlation in gross_correlations.items():
        if feature != 'Gross':
            st.write(f"- **{feature}**: correlação de {correlation:.2f}")
    
    st.write("\nEstes valores são calculados a partir do dataset fornecido e podem variar com os filtros aplicados.")

with st.expander("3. Quais insights podem ser tirados com a coluna Overview?"):
    st.write("""
    - É possível inferir o gênero com boa precisão usando NLP
    - Palavras específicas indicam gêneros:
      * Drama: 'life', 'family', 'story', 'love'
      * Ação: 'action', 'world', 'save', 'mission'
      * Comédia: 'funny', 'comedy', 'laugh', 'hilarious'
    - Overviews mais longas tendem a ser de filmes dramáticos
    - Overviews de ação são mais curtas e diretas
    """)

with st.expander("4. Como prever a nota do IMDB?"):
    st.write("""
    - **Tipo de problema**: Regressão
    - **Modelo escolhido**: Random Forest 
    - **Variáveis importantes**: Meta_score, No_of_Votes, Runtime, Director
    - **Métrica de performance**: MAE (Mean Absolute Error)
    """)

# Tabela de Dados 
st.markdown('---')
st.subheader("🎭 Tabela de Filmes")

if not df_filtrado.empty:
    st.dataframe(df_filtrado[[
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
        'Gross'
    ]])
else:
    st.warning("Nenhum filme encontrado para os filtros selecionados.")

# Esclarecimentos:
st.sidebar.markdown('---')
st.sidebar.info("""
**Desafio Ciência de Dados**
\nEste dashboard permite explorar dados de filmes e realizar análises preditivas.
\nDesenvolvido para o desafio da Indicium.
""")