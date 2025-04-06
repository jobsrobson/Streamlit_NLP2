import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt


# Downloads e configurações iniciais
nltk.download('punkt')
analyzer = SentimentIntensityAnalyzer()

# Carregamento dos dados
df = pd.read_csv('data/tweets.csv', sep=',', encoding='utf-8')

# Função de pré-processamento
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+", "", tweet)  # remove URLs, mentions e hashtags
    # tweet = re.sub(r"[^\w\s]", "", tweet)  # remove pontuação
    tweet = tweet.lower()
    return tweet

# Análise com TextBlob
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

# Análise com VADER
def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# Configurações da página
st.set_page_config(page_title='Análise de Sentimentos de Tweets', layout='wide')


# SIDEBAR
st.sidebar.title('Análise de Sentimentos de Tweets')
st.sidebar.caption('Comparativo entre Bibliotecas Python')

selected_library = st.sidebar.selectbox('Selecione a biblioteca:', ('TextBlob', 'VaderSentiment'))

if selected_library == 'TextBlob':
    library = 'TextBlob'
elif selected_library == 'VaderSentiment':
    library = 'VaderSentiment'

# Estado da análise (usando session_state)
if 'analyze' not in st.session_state:
    st.session_state.analyze = False

# Botão de Analisar
if st.sidebar.button('Realizar Análise'):
    st.session_state.analyze = True
    # st.sidebar.success('Análise realizada com sucesso!')

# Botão de Reset
if st.sidebar.button('Resetar'):
    st.session_state.analyze = False
    st.rerun()

# LÓGICA: só executa se o botão foi clicado
if st.session_state.analyze:
    df['cleaned_text'] = df['text'].apply(clean_tweet)

    if library == 'TextBlob':
        df['sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)
    elif library == 'VaderSentiment':
        df['sentiment'] = df['text'].apply(get_vader_sentiment)

    # MAIN
    st.write(f'### Análise de Sentimentos usando {library}')

    with st.container():
        col1, col2, col3, col4 = st.columns(4, border=True)
        col1.metric(label="Total de Tweets", value=len(df['airline_sentiment']))
        col2.metric(
            label="Tweets Positivos",
            value=len(df[df['sentiment'] == 'positive']),
            delta=len(df[df['sentiment'] == 'positive']) - len(df[df['airline_sentiment'] == 'positive']),
            delta_color="normal"
        )

        col3.metric(
            label="Tweets Negativos",
            value=len(df[df['sentiment'] == 'negative']),
            delta=f"{len(df[df['sentiment'] == 'negative']) - len(df[df['airline_sentiment'] == 'negative'])}",
            delta_color="normal"
        )

        col4.metric(
            label="Tweets Neutros",
            value=len(df[df['sentiment'] == 'neutral']),
            delta=len(df[df['sentiment'] == 'neutral']) - len(df[df['airline_sentiment'] == 'neutral']),
            delta_color="normal"
        )

    # Exibir dataframe
    with st.expander("Visualizar dados analisados"):
        st.dataframe(df[['airline', 'text', 'sentiment']])

    with st.container(border=True):
        # Contagem dos sentimentos
        real_counts = df['airline_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)
        pred_counts = df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)

        # DataFrame para Altair
        comparativo_df = pd.DataFrame({
            'Sentimento': ['Positivo', 'Neutro', 'Negativo'] * 2,
            'Quantidade': list(real_counts) + list(pred_counts),
            'Tipo': ['Real'] * 3 + ['Previsto'] * 3
        })

        # Gráfico de barras com xOffset
        bar_chart = alt.Chart(comparativo_df).mark_bar(size=40).encode(
            x=alt.X('Sentimento:N', title='Sentimento'),
            y=alt.Y('Quantidade:Q', title='Quantidade de Tweets'),
            color=alt.Color('Tipo:N', scale=alt.Scale(range=['#ff7356', '#00d182']), legend=alt.Legend(title="Tipo")),
            xOffset='Tipo:N',
            tooltip=['Sentimento', 'Tipo', 'Quantidade']
        )

        # Rótulos acima das barras
        labels = alt.Chart(comparativo_df).mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # deslocamento vertical
            fontSize=12,
            color='white',
        ).encode(
            x=alt.X('Sentimento:N'),
            y=alt.Y('Quantidade:Q'),
            xOffset='Tipo:N',
            detail='Tipo:N',
            text='Quantidade:Q'
        )

        # Combinação dos dois
        chart = (bar_chart + labels).properties(
            width=300,
            height=400,
            title='Comparação de Sentimentos Reais vs Previstos',
        ).configure_mark(
            opacity=1
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14,
            titleColor='white',
            labelColor='white',
            labelAngle=0
        ).configure_title(
            fontSize=16,
            anchor='start',
            color='white',
            fontWeight='bold'
        ).configure_legend(
            titleFontSize=14,
            labelFontSize=12,
            orient='top-right',
            labelColor='white',
            titleColor='white',
            padding=10,
            cornerRadius=8,
            strokeColor='white'
        ).configure_view(
        )

        st.altair_chart(chart, use_container_width=True)

else:
    st.info('Selecione a biblioteca e clique em **Realizar Análise** para começar.')