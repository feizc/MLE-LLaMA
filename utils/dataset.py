import json 
from torch.utils.data import Dataset

CUTOFF_LEN = 256

class TextDataSet(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.data = json.load(open(file_path, 'r')) 
        self.tokenizer = tokenizer 
    
    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, index):
        texts = self.data[index]['instruction'] + ' ' + self.data[index]['input'] + ' ' + self.data[index]['output'] 
        result = self.tokenizer(texts, 
                                    truncation=True,
                                    max_length=CUTOFF_LEN + 1,
                                    padding="max_length", )
                                    # return_tensors="pt",)
        return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
                }

        
