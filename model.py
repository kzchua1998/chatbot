from transformers import BertModel
import torch

class SimpleNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.dense1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomBERTWithSimpleNN(torch.nn.Module):

    def __init__(self, bert_model, simple_nn):
        super(CustomBERTWithSimpleNN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.simple_nn = simple_nn

    def forward(self, ids, mask):
        # Get BERT representations
        sequence_output, _ = self.bert(ids, attention_mask=mask)
        bert_output = sequence_output[:, 0, :]  # Using the [CLS] token representation

        # Pass BERT output through the trained SimpleNN layers
        simple_nn_output = self.simple_nn(bert_output)

        return simple_nn_output
    

# class CustomBERTModel(torch.nn.Module):

#     def __init__(self):
#           super(CustomBERTModel, self).__init__()
#           self.bert = BertModel.from_pretrained("bert-base-uncased")
#           ### New layers:
#           self.linear1 = torch.nn.Linear(768, 128)
#           self.linear2 = torch.nn.Linear(128, 3) ## 3 is the number of classes in this example


#     def forward(self, ids, mask):
#           sequence_output, pooled_output = self.bert(
#                ids, 
#                attention_mask=mask)

#           # sequence_output has the following shape: (batch_size, sequence_length, 768)
#           linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
#           linear2_output = self.linear2(linear1_output)

#           return linear2_output