# ML-LLaMA: Multi-Language LLaMA

This project aims to make LLaMa understand Chinese, and can generate fluency chinese. 

- [X] Token vocabulary support for multi-language. We found that llama tokenizer naturally support for Chinese. 
- [X] Fine-tuning llama script.  

  (1) original script must be run on 80G A100 and more techniques should be employed. 
  
  (2) lora fine-tuning using [pert](https://github.com/huggingface/peft).
  
- [ ] Fine-grained english-chinese dataset. We are collecting alignment dataset.
- [ ] Instructing tuning


## Reference 
[1] https://github.com/facebookresearch/llama 

[2] https://github.com/tatsu-lab/stanford_alpaca 

[3] https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling

[4] https://github.com/tloen/alpaca-lora
