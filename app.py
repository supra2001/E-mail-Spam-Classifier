import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#st.markdown("<h1 style='color:yellow; text-align:center;'>Email Spam Classifier</h1>", unsafe_allow_html=True)
st.title('Email Spam Classifier')

st.sidebar.title("About")
st.sidebar.info("This app classifies E-mail or SMS messages as Spam or Not Spam")

input_sms = st.text_area("Please Enter the messeage here")

def transforming_text(text):
    
    text = text.lower()
    
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    z = []
    for j in y:
        if j not in stopwords.words('english') and j not in string.punctuation :
            z.append(j)
    
    temp = []
    for word in z:
        temp.append(ps.stem(word))

    return " ".join(temp)

if st.button("Predict"):

    # 1. Preprocessing
    transformed_sms = transforming_text(input_sms)
    # 2. Vectorize
    vectorized_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vectorized_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<h1 style='color:red;'>Spam</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='color:green;'>Not Spam</h1>", unsafe_allow_html=True)
        
        
st.markdown(
    "<footer style='position: fixed; bottom: 50px; right: 10px; font-size:15px; color: black; "
    "background-color:rgb(232, 232, 6); padding: 8px 15px; border-radius: 10px; "
    "text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);'>"
    "Created by Supritam Mukherjee</footer>",
    unsafe_allow_html=True,
)

