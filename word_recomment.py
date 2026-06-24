import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import streamlit as st

st.title("Word Recommendation System")

a = st.text_area("Enter the text to get recommendations :")

bt1 = st.button("Submit")

if bt1:

    p = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
         '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^',
         '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5',
         '6', '7', '8', '9']

    s = ""

    for i in a:
        if i not in p:
            s = s + i

    s = s.lower()

    s1 = sent_tokenize(s)

    l = []
    for i in s1:
        l.append(word_tokenize(i))

    stop_words = stopwords.words('english')

    l2 = []
    for i in l:
        l3 = []
        for j in i:
            if j != '.' and j not in stop_words:
                l3.append(j)
        l2.append(l3)

    ps = PorterStemmer()

    l4 = []
    for i in l2:
        l5 = []
        for j in i:
            l5.append(ps.stem(j))
        l4.append(l5)

    le = WordNetLemmatizer()

    l6 = []
    for i in l4:
        l7 = []
        for j in i:
            l7.append(le.lemmatize(j))
        l6.append(l7)

    model = Word2Vec(
        sentences=l6,
        min_count=1,
        vector_size=100,
        window=5,
        workers=4
    )

    st.session_state["model"] = model
    st.success("Model Trained Successfully!")

if "model" in st.session_state:

    t = st.text_input("Enter a word to get recommendations :")

    bt2 = st.button("Get Recommendations")

    if bt2:

        model = st.session_state["model"]

        if t.lower() in model.wv.key_to_index:

            result = model.wv.most_similar(t.lower(), topn=3)

            st.subheader("Recommended Words")

            for i in range(len(result)):
                st.write(result[i][0])

        else:
            st.error("Word not found in the given text.")