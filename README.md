# analise-dados-filmes
Um dashboard interativo para análise de dados de filmes do IMDb, construído com Streamlit.

📊 Funcionalidades
Filtros interativos: Ano de lançamento, gênero, diretor e certificação

Métricas principais: Nota média, bilheteria total, filme mais bem avaliado

Visualizações gráficas:

Top 10 filmes por avaliação

Relação entre duração e avaliação

Distribuição das avaliações

Evolução temporal das avaliações

Tabela interativa com dados detalhados dos filmes

🛠️ Tecnologias Utilizadas
Python 3.7+

Pandas 2.2.3

Streamlit 1.44.1

Plotly 5.24.1

📋 Pré-requisitos
Antes de executar o projeto, certifique-se de ter instalado:

Python 3.7 ou superior

pip (gerenciador de pacotes do Python)

Como Executar?
1. Clone o repositório:
git clone https://github.com/seu-usuario/analise-dados-filmes-.git 

cd imdb-dashboard

2. Instale as dependências:
pip install -r requirements.txt

3. Execute
streamlit run app.py

4.Acessar o dashboard no navegador:
http://localhost:8501

Estrutura do projeto:
analise-dados-filmes/
├── app.py              # Código principal da aplicação
├── requirements.txt    # Dependências do projeto
├── desafio_indicium_imdb.csv  # Dataset do IMDb
└── README.md          # Este arquivo

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