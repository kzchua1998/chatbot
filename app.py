import pandas as pd
import numpy as np
import pickle
import torch
import yaml

from predict import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# app_data = config["app"]

bert_model, bert_tokenizer, spec_cls_model, device = get_models()
df_processed = get_processed_df(DF_PATH, bert_model, bert_tokenizer, device)

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: InputData):
    
    try:
        user_text_embeddings = get_bert_embeddings(bert_model, bert_tokenizer, input_data.text, device)
        predicted_spec = predict_spec(spec_cls_model, user_text_embeddings, DIFF_THRESHOLD)
       
        df_filtered = df_processed[df_processed["specialization"].isin(predicted_spec)]
        recommended_jobs = predict_job(df_filtered, user_text_embeddings)

        # Return the predictions
        return {"recommended_job": recommended_jobs, "predicted_spec": predicted_spec}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
