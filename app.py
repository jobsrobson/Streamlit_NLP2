import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
from wordcloud import WordCloud
from sklearn.metrics import classification_report


# Configurações da página
st.set_page_config(page_title='Análise de Sentimentos de Tweets', layout='wide')


# Downloads e configurações iniciais
nltk.download('punkt')
analyzer = SentimentIntensityAnalyzer()


# MODAL DE INFORMAÇÕES
@st.dialog("❓ Sobre as bibliotecas de análise de sentimentos", width = "large")
def show_library_info():
        st.markdown("""
   Neste aplicativo, você pode escolher entre duas bibliotecas populares para análise de sentimentos em texto: **TextBlob** e **VADER (Valence Aware Dictionary and sEntiment Reasoner)**.

    ### TextBlob
    - Baseada em **léxico e polaridade**.
    - Calcula a **polaridade** (variação de -1 a +1) e a **subjetividade** do texto.
    - Classificação do sentimento:
      - `Positive` se a polaridade > 0
      - `Negative` se a polaridade < 0
      - `Neutral` se a polaridade = 0
    - É simples e eficaz para textos mais formais ou estruturados.
    - Não leva em conta contexto informal, ironia ou emojis.

    ### VADER
    - Projetada especificamente para **textos de redes sociais**, como tweets, comentários, avaliações, etc.
    - Considera:
      - Emojis
      - Uso de maiúsculas
      - Pontuação excessiva (!!!)
      - Intensificadores e negações
    - Utiliza uma **pontuação composta** chamada `compound` (de -1 a +1) para determinar o sentimento:
      - `Positive` se compound ≥ 0.05
      - `Negative` se compound ≤ -0.05
      - `Neutral` caso contrário
    - Mais sensível a nuances informais e expressividade textual.

    > **Dica:** VADER tende a ser mais assertiva em textos curtos, informais e com linguagem emocional, como tweets. Já o TextBlob é ótimo para análises básicas e rápidas, especialmente em textos neutros ou técnicos.

    """, unsafe_allow_html=True)



# SIDEBAR
st.sidebar.title('💭Comparativo de Ferramentas de Sentimento')
st.sidebar.caption(
    "Avaliação e comparação entre bibliotecas Python de análise de sentimentos "
    "em tweets reais sobre companhias aéreas americanas. Explore os resultados, "
    "visualize os erros e entenda o comportamento de cada ferramenta."
)

col1, col2 = st.sidebar.columns([10, 1], gap="small")

with col1:
    selected_library = st.selectbox(
        'Selecione uma biblioteca',
        ('Selecione...', 'TextBlob', 'VaderSentiment'),
        index=0
    )
   
with col2:
    st.markdown("<div style='margin-top: 1.70rem'></div>", unsafe_allow_html=True)

    if st.button(":material/help:", help="Clique para ver detalhes sobre as bibliotecas" , type="tertiary"):
        show_library_info()


library = selected_library
if selected_library == 'Selecione...':
    st.sidebar.error("Selecione uma biblioteca para continuar.")
    st.write('##### 👈 Selecione uma biblioteca para começar!')
    st.stop()


# Carregamento dos dados
df = pd.read_csv('data/tweets.csv', sep=',', encoding='utf-8')


# FUNÇÕES
# Função de pré-processamento
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|@\S+|#\S+", "", tweet)  # remove mentions e hashtags
    tweet = re.sub(r"t\.co\S+", "", tweet)  # remove URLs t.co
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
    

with st.spinner("🔄 Processando os tweets..."):
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    if library == 'TextBlob':
        df['sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)
    elif library == 'VaderSentiment':
        df['sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)


# ROW 1 - TÍTULO E MÉTRICAS

st.subheader(f'Análise de Sentimentos usando {library}')
st.write('')
with st.container():
    col1, col2, col3, col4 = st.columns(4, border=True)
    col1.metric(
        label="🐤 Total de Tweets", value=f"{len(df['airline_sentiment'])}", 
        help="Total de tweets analisado nesta base de dados")
    
    col2.metric(
        label="😍 Tweets Positivos",
        value=len(df[df['sentiment'] == 'positive']),
        delta=len(df[df['sentiment'] == 'positive']) - len(df[df['airline_sentiment'] == 'positive']),
        delta_color="normal",
        help=f"Total de tweets que o modelo classificou como positivos. \n\n O total real é de {len(df[df['airline_sentiment'] == 'positive'])} tweets positivos."
    )
    col3.metric(
        label="😐 Tweets Neutros",
        value=len(df[df['sentiment'] == 'neutral']),
        delta=len(df[df['sentiment'] == 'neutral']) - len(df[df['airline_sentiment'] == 'neutral']),
        delta_color="normal",
        help=f"Total de tweets que o modelo classificou como neutros. \n\n O total real é de {len(df[df['airline_sentiment'] == 'neutral'])} tweets neutros."
    )
    col4.metric(
        label="😡 Tweets Negativos",
        value=len(df[df['sentiment'] == 'negative']),
        delta=f"{len(df[df['sentiment'] == 'negative']) - len(df[df['airline_sentiment'] == 'negative'])}",
        delta_color="normal",
        help=f"Total de tweets que o modelo classificou como negativos. \n\n O total real é de {len(df[df['airline_sentiment'] == 'negative'])} tweets negativos."
    )


# ROW 2 - COMPARATIVO DO MODELO

col1, col2 = st.columns([3,2], gap="small")

with col1:
    with st.container(border=True, height=341):
        
        # Contagem dos sentimentos
        real_counts = df['airline_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)
        pred_counts = df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'], fill_value=0)

        comparativo_df = pd.DataFrame({
            'Sentimento': ['Positivo', 'Neutro', 'Negativo'] * 2, 
            'Quantidade': list(real_counts) + list(pred_counts),
            'Tipo': ['Real'] * 3 + ['Previsto'] * 3
        })

        from pandas.api.types import CategoricalDtype

        ordem_sentimentos = CategoricalDtype(categories=["Positivo", "Neutro", "Negativo"], ordered=True)
        comparativo_df["Sentimento"] = comparativo_df["Sentimento"].astype(ordem_sentimentos)

        # Altair chart
        bar_chart = alt.Chart(comparativo_df).mark_bar(size=40).encode(
            x = alt.X('Sentimento:N', title=None),
            y = alt.Y('Quantidade:Q', title='Quantidade de Tweets'),
            color = alt.Color('Tipo:N', scale=alt.Scale(range=['#ff7356', '#00d182']), legend=alt.Legend(title="Tipo")),
            xOffset = 'Tipo:N',
            tooltip = ['Sentimento', 'Tipo', 'Quantidade']
        )
        
        # Gráfico de barras
        bar_chart = alt.Chart(comparativo_df).mark_bar(size=40).encode(
            x = alt.X('Sentimento:N', title=None, sort=['Positivo', 'Neutro', 'Negativo']),
            y = alt.Y('Quantidade:Q', title='Quantidade de Tweets'),
            color = alt.Color('Tipo:N', scale=alt.Scale(range=['#ff7356', '#00d182']), legend=alt.Legend(title="Tipo")),
            xOffset = 'Tipo:N',
            tooltip = ['Sentimento', 'Tipo', 'Quantidade']
        )
        labels = alt.Chart(comparativo_df).mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=12,
            color='white',
        ).encode(
            x=alt.X('Sentimento:N'),
            y=alt.Y('Quantidade:Q'),
            xOffset='Tipo:N',
            detail='Tipo:N',
            text='Quantidade:Q'
        )
        chart = (bar_chart + labels).properties(
            height=295,
            title='Comparação de Sentimentos Reais vs Previstos',
            ).configure_mark(
                opacity=1
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14,
                labelAngle=0
            ).configure_title(
                fontSize=16,
                anchor='start',
                fontWeight='bold'
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12,
                orient='top-right',
                padding=10,
                cornerRadius=8,
            ).configure_view(
        )
        st.altair_chart(chart, use_container_width=True)
    
with col2:
    with st.container(border=True):
        from sklearn.metrics import accuracy_score, confusion_matrix
        # Acurácia
        accuracy = accuracy_score(df['airline_sentiment'], df['sentiment'])
        st.metric("🎯 Acurácia do Modelo", f"{accuracy * 100:.2f}%", help="Percentual de classificações corretas.")
    
    with st.container(border=True, height=220):
        # Matriz de confusão
        conf_matrix = confusion_matrix(df['airline_sentiment'], df['sentiment'], labels=['positive', 'neutral', 'negative'])
        conf_df = pd.DataFrame(conf_matrix,
                            index=['Real: Positivo', 'Real: Neutro', 'Real: Negativo'],
                            columns=['Previsto: Positivo', 'Previsto: Neutro', 'Previsto: Negativo'])
        st.caption("😵‍💫 Matriz de Confusão")
        st.dataframe(conf_df, use_container_width=True)


# ROW 3 - ERROS E RELATÓRIO

col1, col2 = st.columns([3,2], gap="small")

with col1:
    with st.container(border=True):
        st.caption("❌ Exemplos em que o modelo errou")
        erros = df[df['airline_sentiment'] != df['sentiment']]
        st.dataframe(erros[['airline', 'text', 'airline_sentiment', 'sentiment']].head(20), use_container_width=True, height=142, hide_index=True)

with col2:
    with st.container(border=True):
        # Relatório de classificação
        report = classification_report(df['airline_sentiment'], df['sentiment'], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(2)
        
        st.caption("📊 Relatório de desempenho por classe")
        st.dataframe(report_df.loc[["positive", "neutral", "negative"]][["precision", "recall", "f1-score"]])
        
 
# ROW 4 - NUVENS DE PALAVRAS

from PIL import Image

nomes_sentimentos = {
    "positive": "Positivos",
    "neutral": "Neutros",
    "negative": "Negativos"
}

with st.container(border=True):
    col1, col2, col3 = st.columns(3, gap="medium")

    for col, sent, color in zip(
        [col1, col2, col3],
        ['positive', 'neutral', 'negative'],
        ['Greens', 'Blues', 'Reds']
    ):
        text = " ".join(df[df['airline_sentiment'] == sent]['text'])

        if text.strip():
            wordcloud = WordCloud(
                width=500,
                height=300,
                mode="RGBA",  # permite transparência real
                background_color=None,
                colormap=color,
                max_words=20,
                font_path="fonts/INST_SANS_BOLD.ttf"
            ).generate(text)

            image = wordcloud.to_image()

            with col:
                col.markdown(f"**{nomes_sentimentos[sent]}**")
                st.image(image, use_container_width=True)
        else:
            with col:
                col.markdown(f"**{nomes_sentimentos[sent]}**")
                st.error("Sem palavras suficientes para gerar a nuvem.")


# ROW 5

with st.expander("👁️ Visualizar dados analisados"):
    st.dataframe(df[['airline', 'text', 'sentiment', 'airline_sentiment']], use_container_width=True)


# FOOTER
st.caption('🧑‍💻 Made by [**Robson Ricardo**](https://github.com/jobsrobson)')