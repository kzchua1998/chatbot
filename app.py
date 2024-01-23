import pandas as pd
import numpy as np
import pickle
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

BERT_MODEL_PATH = "bert-base-uncased"

app = FastAPI()

class InputData(BaseModel):
    text: str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained(BERT_MODEL_PATH).to(device)
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

df = pd.read_csv(r"mini-job-rec-dataset - jobs_data.csv").dropna()
df["specialization"] = df["specialization"].apply(lambda x: "IT & Software Developer" if x in ["IT","Software Developer"] else x)

# Vectorize job descriptions
job_vectors = list()
job_specs = list()
job_titles = list()

for title, spec, description, skillset in zip(df["job_title"],df["specialization"],df['job_description'],df['skillsets']):
    inputs = bert_tokenizer(description + " " + skillset, return_tensors="pt").to(device)
    outputs = bert_model(**inputs)
    vector_representation = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    job_vectors.append(vector_representation)
    job_specs.append(spec)
    job_titles.append(title)

with open('spec_cls.pkl', 'rb') as file:
    spec_cls_model = pickle.load(file)


@app.post("/predict")
async def predict(input_data: InputData):
    
    try:
        # Process input text through BERT
        with torch.no_grad():
            tokenized_input = bert_tokenizer(input_data.text, return_tensors="pt").to(device)
            outputs = bert_model(**tokenized_input)
            user_bert_features = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

        # Make predictions using the machine learning model
        predictions = spec_cls_model.predict_proba([user_bert_features])[0]
        top2_spec = predictions.argsort()[::-1][:2]
        proba_difference = predictions[top2_spec[0]] - predictions[top2_spec[1]]

        if proba_difference >= 0.15:
            predicted_spec = [spec_cls_model.classes_[np.argmax(predictions)]]
        else:
            predicted_spec = [spec_cls_model.classes_[top2_idx] for top2_idx in top2_spec]

        job_vectors_filtered = [job_vector for job_vector,job_spec in zip(job_vectors,job_specs) if job_spec in predicted_spec]
        job_titles_filtered = [title for title,job_spec in zip(job_titles,job_specs) if job_spec in predicted_spec]

        # Calculate cosine similarity between user input vector and job description vectors
        similarities = cosine_similarity(user_bert_features.reshape(1, -1), job_vectors_filtered)

        # Get indices of top-N most similar job descriptions
        N = 3  # Adjust N based on the number of recommendations you want to provide
        top_n_indices = similarities.argsort()[0][::-1][:N]
        print(predictions)
        # Recommend top-N jobs
        # recommended_jobs = df.iloc[top_n_indices]["job_title"]
        recommended_jobs = [job_titles_filtered[idx] for idx in top_n_indices]

        # Return the predictions
        return {"recommended_job": [job for job in recommended_jobs], "predicted_spec": predicted_spec}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
