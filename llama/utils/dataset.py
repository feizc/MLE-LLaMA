import json 
from torch.utils.data import Dataset

class TextDataSet(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.data = json.load(open(file_path, 'r')) 
        self.tokenizer = tokenizer 
    
    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, index):
        texts = self.data[index]['instruction'] + ' ' + self.data[index]['input'] + ' ' + self.data[index]['output'] 
        texts_ids = self.tokenizer(texts, return_tensors="pt",)
        return texts_ids
        
