#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb


# ## Time

# In[3]:


import time
start=time.time()


# ## Data

# ###### 1. Import Data

# In[4]:


df = pd.read_csv('Book5.csv', header=None)


# ###### 2. Take sentences 

# In[5]:


batch_1 = df[:300]


# In[5]:


#We can see how many sentences are positive and negative
batch_1[1].value_counts()


# ## Loading the Pre-trained BERT model 

# In[6]:


# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

#for Bert
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')


# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# ## Put in Cuda

# In[1]:


device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)


# ## Preparing the Dataset

# ###### 1. Tokenization 

# In[8]:


#Break sentences into word and subwords for Bert
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# ###### 2. Padding 

# In[9]:


max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


# In[10]:


np.array(padded).shape


# In[11]:


type(padded)


# ###### 3. Masking

# In[12]:


attention_mask = np.where(padded != 0, 1, 0)


# # Model 1 Deeplearning

# ###### The model() function runs our senstences through BERT.
#     The results of the processing will be returned into Last_hidden_states
#     

# In[13]:


model.to(device)


# In[14]:


input_ids = torch.tensor(padded) 
input_ids.to(device)


# In[15]:


attention_mask = torch.tensor(attention_mask)
attention_mask.to(device)


# In[16]:


with torch.no_grad():
    last_hidden_states = model(input_ids.cuda(), attention_mask=attention_mask.cuda())


# In[17]:


input_ids.shape


# In[18]:


input_ids


# In[19]:


features = last_hidden_states[0][:,0,:].cpu()
features.shape


# # Model Train/Split

# In[20]:


labels = batch_1[1]
labels


# ##### Split the data 

# In[21]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
test_features.shape


# In[54]:


lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)


# # Evaluate model

# ###### Check the accuracy against the testing data

# In[55]:


lr_clf.score(test_features, test_labels)


# In[45]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # Test  a sentence

# In[25]:


df2 = pd.read_csv('testchamikara.csv', header=None)


# In[26]:


batch_2 = df2


# In[27]:


#Break sentences into word and subwords for Bert
tokenized2= df2[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized2


# In[28]:


padded2=tokenized2


# In[29]:


attention_mask2 = np.where(padded2 != 0, 1, 0)
attention_mask2


# In[30]:


labels2=df2[1]
labels2


# In[31]:


#Put the model back in the CPU just for one sentence
model.to("cpu")


# In[32]:


input_ids2 = torch.tensor(tokenized2)  
attention_mask2=torch.tensor(attention_mask2)
with torch.no_grad():
    last_hidden_states2=model(input_ids2)
last_hidden_states2


# In[33]:


features2=last_hidden_states2[0][:,0,:].numpy()


# In[34]:


lr_clf.decision_function(features2)


# ## End Time

# In[35]:


end=time.time()
print(end-start)


# In[ ]:




