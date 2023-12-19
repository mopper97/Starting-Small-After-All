from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1

# 1. Load the Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("tokenizer/", max_len=512)

# 2. Define Function for Loading Dataset
def load_dataset(train_path, tokenizer):
    dataset = []
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip('\n')
            tokenized_line = tokenizer.encode(line, return_tensors='pt')[0]
            if len(tokenized_line) > 4:
                dataset.append({'input_ids': tokenized_line})
    return dataset

# 3. Load the Dataset(s)
train_dataset = load_dataset("../../data/aochildes.train", tokenizer)

# eval_dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path="path_to_your_eval_dataset.txt",  # Replace with the path to your eval dataset
#     block_size=128,  # Replace with the block size that suits your data
# )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

# 4. Configure the Model
config = RobertaConfig(
vocab_size=30_000,  # we align this to the tokenizer vocab_size
max_position_embeddings=514,
hidden_size=128,
num_attention_heads=2,
num_hidden_layers=2,
type_vocab_size=1,
intermediate_size=512,
position_embedding_type='relative_key_query')


model = RobertaForMaskedLM(config=config)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./output",  # Replace with the path where you want to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Replace with the number of epochs you want
    per_device_train_batch_size=128,  # Replace with your training batch size
    save_steps=10_000,
    save_total_limit=1,
)

# 6. Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("./output")  # Replace with the path where you want to save the model
