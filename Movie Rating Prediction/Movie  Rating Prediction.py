#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[8]:


df_movies = pd.read_csv("tmdb_5000_movies.csv")
df_credits = pd.read_csv("tmdb_5000_credits.csv")


# In[9]:


df_movies.head(1)


# In[10]:


df_credits.head(1)


# In[11]:


df_movies = df_movies.merge(df_credits, on = "title")


# In[12]:


df_movies.info()


# In[13]:


df_movies = df_movies[["movie_id","title","genres","keywords","overview","cast","crew"]]


# In[14]:


df_movies.info()


# In[15]:


df_movies.head()


# In[16]:


df_movies.isnull().sum()


# In[17]:


df_movies.dropna(inplace = True)


# In[18]:


df_movies.duplicated().sum()


# In[19]:


import ast
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i["name"])
    return l


# In[20]:


df_movies["genres"] = df_movies["genres"].apply(convert)
df_movies["keywords"] = df_movies["keywords"].apply(convert)


# In[21]:


df_movies.head()


# In[22]:


def fetch_cast(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i["name"])
            counter+=1
        else:
            break
    return l


# In[23]:


df_movies["cast"] = df_movies["cast"].apply(fetch_cast)


# In[24]:


df_movies.head()


# In[25]:


def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            l.append(i["name"])
            break
    return l


# In[26]:


df_movies["crew"] = df_movies["crew"].apply(fetch_director)


# In[27]:


df_movies.head()


# In[28]:


df_movies["overview"] = df_movies["overview"].apply(lambda x:x.split())


# In[29]:


df_movies.head()


# In[30]:


df_movies["genres"] = df_movies["genres"].apply(lambda x:[i.replace(" ","") for i in x])
df_movies["keywords"] = df_movies["keywords"].apply(lambda x:[i.replace(" ","") for i in x])
df_movies["cast"] = df_movies["cast"].apply(lambda x:[i.replace(" ","") for i in x])
df_movies["crew"] = df_movies["crew"].apply(lambda x:[i.replace(" ","") for i in x])


# In[31]:


df_movies.head()


# In[32]:


df_movies["tags"] = df_movies["overview"] + df_movies["genres"] + df_movies["keywords"] + df_movies["cast"] + df_movies["crew"]


# In[33]:


df_movies.head()


# In[34]:


new_df = df_movies[["movie_id","title","tags"]]


# In[56]:


new_df["tags"] = new_df["tags"].apply(lambda x:" ".join(x))


# In[57]:


new_df.head()


# In[58]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[39]:


def stem(text):
    y = []
    for i in text.split():
         y.append(ps.stem(i)) 
    return " ".join(y)


# In[55]:


new_df["tags"] = new_df["tags"].apply(stem)


# In[44]:


new_df.head()


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words="english")


# In[47]:


vectors = cv.fit_transform(new_df["tags"]).toarray()


# In[48]:


vectors.shape


# In[49]:


vectors


# In[50]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[51]:


def recommend(movie):
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True,key = lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[52]:


recommend("Batman Begins")


# In[ ]:




