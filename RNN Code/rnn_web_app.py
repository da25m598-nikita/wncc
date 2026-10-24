import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pypickle

st.title("Movie Review Sentinant Analysis")

words_dic = pypickle.load("word_token_ids.pkl")

def convert_text_into_word_id(text):
   text = text.lower().split()
   print(text)
   word_ids = [1]
   for t in text:
      if t == "?":
         word_ids.append(2)
      else:
         word_id = words_dic.get(t,2)
         if word_id != 2:
               word_ids.append(word_id+3)
         else:
              word_ids.append(word_id)

   return word_ids


@st.cache_resource
def load_rnn_model():
    model = load_model("movie_review_detection.keras")
    return model


review = st.text_area("Moview Review",placeholder="Write your review")

isClicked = st.button("Predict",type='primary')

if isClicked:
    word_ids = convert_text_into_word_id(review)
    word_ids = pad_sequences([word_ids],padding='pre',truncating='post',maxlen=200)
    rnn = load_rnn_model()
    pred = rnn.predict(word_ids)
    with st.spinner("Processing......."):
        if pred >= 0.5:
            st.image("./positive.jpg",width=200)
            st.success("Positive Review")
        else:
            st.image("./negative.jpg",width=200)
            st.error("Negative Review")












