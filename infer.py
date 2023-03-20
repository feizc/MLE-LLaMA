from llama import LlamaTokenizer, LlamaForCausalLM


ckpt_path = './ckpt'
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
model = LlamaForCausalLM.from_pretrained(ckpt_path) 



input = '你好, 看起来今天的天气很不错，你觉得呢？'
id = tokenizer.encode(input) 
print(id)
print(tokenizer.decode(id))


prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


