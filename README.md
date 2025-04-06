# ✈️ Análise de Sentimentos em Tweets sobre Companhias Aéreas

Este projeto desenvolve um webapp interativo com **Streamlit** para análise de sentimentos em tweets sobre companhias aéreas americanas, utilizando as bibliotecas **TextBlob** e **VADER**.  O principal objetivo é **comparar os resultados obtidos pelas bibliotecas com os rótulos reais do dataset** e avaliar a performance de cada uma.



## 🔍 Sobre

- Selecione uma das bibliotecas de análise de sentimentos (`TextBlob` ou `VADER`)
- O app processa os tweets em tempo real e prevê o sentimento (positivo, neutro ou negativo)
- Compare os resultados previstos com os sentimentos reais do dataset
- Visualize métricas como **acurácia**, **matriz de confusão**, **relatório de classificação**
- Veja **nuvens de palavras** separadas por sentimento
- Teste **um tweet personalizado** para ver a classificação ao vivo (em breve)


## 📊 Tecnologias e Bibliotecas

- [Streamlit](https://streamlit.io/) — para interface web interativa
- [Pandas](https://pandas.pydata.org/) — para manipulação de dados
- [TextBlob](https://textblob.readthedocs.io/en/dev/) — análise de sentimentos léxica
- [VADER](https://github.com/cjhutto/vaderSentiment) — análise de sentimentos voltada a redes sociais
- [Altair](https://altair-viz.github.io/) — visualizações interativas
- [WordCloud](https://amueller.github.io/word_cloud/) — nuvens de palavras
- [Scikit-learn](https://scikit-learn.org/) — métricas de avaliação



## 🧠 Dataset

O dataset utilizado foi retirado do repositório:

📁 [`tweets_airlines`](https://github.com/alexvaroz/data_science_alem_do_basico/raw/refs/heads/master/tweets_airlines.zip)

Inclui milhares de tweets previamente rotulados como `positive`, `neutral` e `negative`.

