# analise-dados-filmes
Um dashboard interativo para análise de dados de filmes do IMDb, construído com Streamlit.

📊 Funcionalidades
Filtros interativos: Ano de lançamento, gênero, diretor e certificação

Métricas principais: Nota média, bilheteria total, filme mais bem avaliado, diretor mais frequente e filme mais popular

Visualizações gráficas:

Top 10 filmes por avaliação

Relação entre duração e avaliação

Distribuição das avaliações

Evolução temporal das avaliações

Análise de texto (NLP): Nuvem de palavras e análise de palavras mais frequentes nas sinopses (overviews) dos filmes

Modelo preditivo: Previsão de notas IMDB usando Random Forest

Tabela interativa com dados detalhados dos filmes

🛠️ Tecnologias Utilizadas
Python 3.7+

Pandas

Streamlit

Plotly

Scikit-learn

Matplotlib

Seaborn

WordCloud

📋 Pré-requisitos
Antes de executar o projeto, certifique-se de ter instalado:

Python 3.7 ou superior

pip (gerenciador de pacotes do Python)

Como Executar?
1. Clone o repositório:
git clone https://github.com/seu-usuario/analise-dados-filmes-.git 

cd analise-dados-filmes

2. Instale as dependências:
pip install -r requirements.txt

3. Execute esse comando para visualizar o dashboard:
streamlit run app.py

4.Acessar o dashboard no navegador:
http://localhost:8501

Como usar?
1.Utilize os filtros na barra lateral para selecionar:

Intervalo de anos

Gêneros de filmes

Diretores específicos

Classificações indicativas

2.Explore as métricas principais no topo do dashboard

3.Interaja com os gráficos para obter insights sobre:

Os filmes melhor avaliados

Relação entre duração e avaliação

Distribuição das notas

Evolução das avaliações ao longo do tempo

4.Consulte a tabela detalhada na parte inferior para ver todos os filmes filtrados

📊 Dataset
O projeto utiliza o dataset desafio_indicium_imdb.csv.


