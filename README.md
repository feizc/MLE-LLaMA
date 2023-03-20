<p align="center">
     <img src="figures/logo.png" alt="logo" width = "550">
     <br/>
</p> 

# MLE-LLaMA: Multi-Language Enhanced LLaMA

This project aims to make LLaMa understand Chinese, and can generate fluency chinese. We are inspired that LLaMa have learned good English expression and a little alignment prompt can makes it capture Chinese. 

- [X] Token vocabulary support for multi-language. We found that llama tokenizer naturally support for Chinese. 
- [X] Fine-tuning llama script.  

  (1) ```train.py``` original script must be run on 80G A100 and more techniques should be employed. 
  
  (2) ```train_lora.py``` lora fine-tuning using [pert](https://github.com/huggingface/peft). 
  
  | Argument | Values |
  |------|------|
  | `batch size` | 128 * 8 |
   | `epochs` | 3 |
   | `cut length` | 256 |
   | `learning rate` | 2e-5 |
   | `speed` | 1.02s / it |
  
  
- [ ] Fine-grained english-chinese dataset. We are collecting alignment dataset.
- [ ] Instructing tuning. We use [chinese alpaca](https://github.com/carbonz0/alpaca-chinese-dataset) and [GuanacoDataset
](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) for instructing tunning. 
- [ ] Checkpoints and cases


## Reference 
[1] https://github.com/facebookresearch/llama 

[2] https://github.com/tatsu-lab/stanford_alpaca 

[3] https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling

[4] https://github.com/tloen/alpaca-lora
