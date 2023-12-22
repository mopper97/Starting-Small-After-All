from tokenizers import ByteLevelBPETokenizer
import os 

input_files = os.listdir("../../data")
print(input_files)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[os.path.join('../../data', x) for x in input_files], vocab_size=30_000, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<mask>'])
tokenizer.save_model("tokenizer/")
