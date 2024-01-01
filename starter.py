import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2 as pyp
import numpy as np

model = SentenceTransformer('bert-base-uncased')

with open('Outlook.pdf', 'rb') as file:
    reader = pyp.PdfReader(file)
    txt = "".join(list(reader.pages[i].extract_text() for i in range(len(reader.pages)-1)))

txt = txt.split('\n\n')
length_threshold = np.percentile(np.array(list(len(i) for i in txt)), 90)
txt = list(para for para in txt if len(para)>=length_threshold)
test_text = model.encode(txt)
test_topic = model.encode("equities")
cosine_sim = np.concatenate(cosine_similarity(test_text, [test_topic]))
top = np.percentile(cosine_sim, 90)

for i in range(len(cosine_sim)):
    if cosine_sim[i]>=top:
        print(txt[i] + '\n')