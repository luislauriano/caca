import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Configuração da página
st.set_page_config(layout="wide")

# ''' Backend '''

# Carregar dados e modelos previamente treinados
articles_dict = pickle.load(open("articles.pkl", "rb"))  # Substitua pelo seu arquivo
articles = pd.DataFrame(articles_dict)

articles.drop_duplicates(inplace=True)

# Carregar matriz TF-IDF e modelo
with open('csr_data_tf.pkl', 'rb') as file:
    csr_data = pickle.load(file)

model = pickle.load(open("model_teste.pkl", "rb"))

# Função para recomendar artigos
def recommend(article_title):
    n_articles_to_recommend = 5
    idx = articles[articles['title'] == article_title].index[0]

    distances, indices = model.kneighbors(csr_data[idx], n_neighbors=n_articles_to_recommend + 1)
    idx = list(indices.squeeze())
    df = np.take(articles, idx, axis=0)

    recommended_titles = list(df.title[1:])
    recommended_ids = list(df.id[1:])

    return recommended_titles, recommended_ids

# Função para buscar metadados dos artigos (exemplo)
def fetch_metadata(article_id):
    article_info = articles[articles["id"] == article_id].iloc[0]
    # Simula busca de metadados do artigo (substituir pela sua lógica ou API real)
    return {
        "abstract": article_info["content"],
        "authors": article_info["authors"]
    }

# ''' Frontend '''

st.write("""<h2><b style="color:red">Cacá - Aba Novidades</b></h2>""", unsafe_allow_html=True)
st.write("###")

st.write("""<p>Bem-vindo ao <b style="color:red">Cacá</b>, um sistema gratuito de recomendação de artigos baseado no seu perfil e no que você tem lido e interagido com a IA.</p>""", unsafe_allow_html=True)
st.write("##")

my_expander = st.expander("Selecione um artigo que você leu para receber recomendação de artigos novos que foram publicados 📚")
selected_article_title = my_expander.selectbox("", articles["title"].values)

def fetch_article_info(article_id):
    article_info = articles[articles["id"] == article_id].iloc[0]
    return {
        "rating": article_info["rating"],
        "votes": article_info["votes"]
    }

def fetch_image_url(article_id):
    article_info = articles[articles["id"] == article_id].iloc[0]
    return article_info["image_url"]



if my_expander.button("Recomendar"):
    st.text("Aqui estão alguns artigos recomendados para você...")
    st.write("#")
    titles, ids = recommend(selected_article_title)
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i in range(0, len(titles)):
        with cols[i]:
            st.write(f' <b style="color:#E50914">{titles[i]}</b>', unsafe_allow_html=True)
            metadata = fetch_metadata(ids[i])
            st.caption(metadata["abstract"])
            st.caption(f"Autores: {(metadata['authors'])}")
            st.write("________")
            
           
            
            # Busca informações adicionais
            article_info = fetch_article_info(ids[i])
            st.write(f'<b style="color:#DB4437">Avaliação:</b> <b>{article_info["rating"]}</b>', unsafe_allow_html=True)
            st.write(f'<b style="color:#DB4437">Citações:</b> <b>{article_info["votes"]}</b>', unsafe_allow_html=True)


st.write("##")
tab1, tab2 = st.tabs(["Sobre", "Funcionamento"])

with tab1:
    st.caption("Este é um sistema de recomendação de artigos baseado em conteúdo usando TF-IDF, com base no seu perfil.")
    st.caption("Em futuras versões, novos artigos serão adicionados. 🚀")
with tab2:
    st.caption("O modelo utiliza TF-IDF para transformar o texto dos artigos em vetores numéricos que representam a relevância das palavras. Em seguida, o Nearest Neighbors calcula a similaridade entre esses vetores usando a métrica de cosseno, permitindo recomendar artigos mais próximos ao selecionado. É uma abordagem baseada em conteúdo, que identifica itens semelhantes com base no texto dos próprios artigos.")
    st.caption("Para mais informações, visite o repositório do projeto.")

