import torch
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

predict_data = config["predict"]
BERT_MODEL_PATH = predict_data["bert_model_path"]
SPEC_CLS_PATH = predict_data["spec_cls_path"]
DF_PATH = predict_data["df_path"]
DIFF_THRESHOLD = predict_data["diff_threshold"]


def get_models():
    # Loading bert and spec classification models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = BertModel.from_pretrained(BERT_MODEL_PATH).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

    with open(SPEC_CLS_PATH, 'rb') as file:
        spec_cls_model = pickle.load(file) 
    return bert_model, bert_tokenizer, spec_cls_model, device


def get_processed_df(DF_PATH, bert_model, bert_tokenizer, device):
    df = pd.read_csv(DF_PATH)
    # # only interested in minimum years of experience
    # df['min_year_of_experience'] = df['year_of_experience'].apply(lambda x: int(x.split('-')[0]) if '-' in x else int(x.replace('+', '')))
    # # add combine_description column for training specialization classifier
    # df["combined_description"] = df['job_title'] + " " + df['job_description'] + " " + df['skillsets']
    # # combine IT and Software Developer into IT & Software Developer column
    # df["specialization"] = df["specialization"].apply(lambda x: "IT & Software Developer" if x in ["IT", "Software Developer"] else x)
    # append bert-embedded description
    df["embedded_combined_description"] = df['combined_description'].apply(lambda x: get_bert_embeddings(bert_model,bert_tokenizer,x,device))
    return df


def get_bert_embeddings(bert_model, bert_tokenizer, text, device):
    with torch.no_grad():
        tokenized_text = bert_tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
        # outputs = bert_model(torch.LongTensor([token]))
        outputs = bert_model(tokenized_text)
        last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    return last_hidden_states


def get_top_n_indices(array, n):
    top_n_indices = array.argsort()[::-1][:n]
    return top_n_indices


def predict_spec(spec_cls_model, bert_embedding, diff_threshold):
    predictions = spec_cls_model.predict_proba([bert_embedding])[0]
    top_2_indices = get_top_n_indices(predictions, 2)
    proba_difference = predictions[top_2_indices[0]] - predictions[top_2_indices[1]]
    if proba_difference >= diff_threshold:
        return [spec_cls_model.classes_[top_2_indices[0]]]
    return [spec_cls_model.classes_[idx] for idx in top_2_indices]


def predict_job(df_filtered, user_embeddings):
    similarities = cosine_similarity(user_embeddings.reshape(1, -1), np.array(df_filtered["embedded_combined_description"].to_list()))[0]
    top_3_indices = get_top_n_indices(similarities, 3)
    recommended_jobs = [(df_filtered["job_title"].iloc[idx], df_filtered["job_description"].iloc[idx]) for idx in top_3_indices]
    return recommended_jobs


if __name__ == '__main__':

    bert_model, bert_tokenizer, spec_cls_model, device = get_models()
    df_processed = get_processed_df(DF_PATH, bert_model, bert_tokenizer, device)

    user_text = "I am actively involved in recruitment by preparing job descriptions, posting ads and managing the hiring process"
    user_text_embeddings = get_bert_embeddings(bert_model, bert_tokenizer, user_text, device)
    predicted_spec = predict_spec(spec_cls_model, user_text_embeddings, DIFF_THRESHOLD)

    df_filtered = df_processed[df_processed["specialization"].isin(predicted_spec)]
    recommended_jobs = predict_job(df_filtered, user_text_embeddings)
    
    print(f"User: {user_text}\n")
    print("Bot: Here are 3 recommended jobs for you:")
    for job, description in recommended_jobs:
        print(f"{job} - {description}")

