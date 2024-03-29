# Job Recommendation Chatbot
This project features a chatbot powered by the `bert-base-uncased` language model, designed for personalized job recommendations with user prompt.

# Demo
https://github.com/kzchua1998/chatbot/assets/64066100/53d9c8c7-1aa8-4684-b547-a578592c3c87

# Prepare the environment

1. Install python requirements.

   ``` shell
   pip install -r requirements.txt
   ```

2. Download the model [`bert-base-uncased`](https://drive.google.com/file/d/17cv-31VHBgKyqZDBzhaX-FL3xeqQCa2f/view?usp=sharing) and extract to models folder in the repository.

   ``` shell
   - chatbot
      - models
         - spec_cls.pkl
         - bert-base-uncased
   ```



# Run the Program

You can run a sanity check first to make sure all packages are installed properly before running the program

#### Sanity Check

``` shell
python predict.py
```
You can expect three job recommendations printed along with their job descriptions

#### Run Chatbot

``` shell
python main.py
```
Note: The chatbot requires a minimum of `10 words` in the description to provide recommendations.

# Future Improvements
#### Performance Optimization
The `bert-base-uncased` model used for text embeddings can be further quantized and optimized using Hugging Face [`Optimum`](https://www.philschmid.de/optimizing-transformers-with-optimum) for inference.

#### Implementation of LLM
LLM-powered chatbot can streamline customer interactions, automate responses, and enhance user engagement with generalizability, intelligent language understanding and generation capabilities.
