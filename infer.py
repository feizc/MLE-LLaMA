from llama import LlamaTokenizer, LlamaForCausalLM


ckpt_path = './ckpt'
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
# model = LLaMAForCausalLM.from_pretrained(ckpt_path) 



input = '你好, 看起来今天的天气很不错，你觉得呢？一些偏僻的字尝试: 很多生僻字属于異體字，其字意思与相应的现代常用字相通，甚至读音（至少中古汉语发音）也相同，是一个字的不同写法，比如“槍”和“鎗”、“裤”和“袴”、“碰”和“掽”、“磷”和“燐”、“坡”和“陂”等。这类生僻字通常是人名、地名或某些古書中出現過的字，一般的字典因为篇幅限制都不會收錄這些字。早期的電腦漢字編碼字库（比如GB 2312）也不能覆蓋所有生僻字，因此經常產生罕見人名、地名用字無法錄入的情況，使得用户不得不借用读音相近的汉字代替。'
id = tokenizer.encode(input) 
print(id)
print(tokenizer.decode(id))

'''
prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
'''