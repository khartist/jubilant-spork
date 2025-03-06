from torch import nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        if model_name == "phobert-base":
            self.model = AutoModel.from_pretrained('vinai/phobert-base')
        elif model_name == "sentence_transformer":
            self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state.mean(dim=1)

if __name__ == "__main__":

    model = TextEncoder()
    print("Model loaded successfully")
    input_ids = TextEncoder.tokenizer(["Cô giáo đang ăn kem", "Chị gái đang thử món thịt dê"], return_tensors='pt', padding=True)['input_ids']
    attention_mask = TextEncoder.tokenizer(["Cô giáo đang ăn kem", "Chị gái đang thử món thịt dê"], return_tensors='pt', padding=True)['attention_mask']
    print(model(input_ids, attention_mask).shape)
