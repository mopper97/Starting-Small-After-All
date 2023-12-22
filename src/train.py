from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from transformers import  DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm
from argparse import ArgumentParser




# 2. Define Function for Loading Dataset
def load_dataset(train_paths, tokenizer):
    dataset = []
    for path in train_paths:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip('\n')
                tokenized_line = tokenizer.encode(line, return_tensors='pt')[0]
                if 512 > len(tokenized_line) > 4:
                    dataset.append({'input_ids': tokenized_line})
    return dataset

def main(args):
    # 1. Load the Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("tokenizer/", max_len=512, truncation=True)

    # 3. Load the Dataset(s)
    train_dataset = load_dataset(["../data/aochildes.train", "../data/bnc_spoken.train", "../data/cbt.train", "../data/children_stories.train", 
                                  "../data/gutenberg.train", "../data/open_subtitles.train", "../data/qed.train", "../data/simple_wikipedia.train",
                                  "../data/switchboard.train", "../data/wiki103.train"], tokenizer)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # 4. Configure the Model
    config = RobertaConfig(
    vocab_size=30_000,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=args.hs,
    num_attention_heads=args.na,
    num_hidden_layers=args.nl,
    type_vocab_size=1,
    intermediate_size=args.ffn,
    position_embedding_type='relative_key_query')


    model = RobertaForMaskedLM(config=config)

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.save_path,  # Replace with the path where you want to save the model
        overwrite_output_dir=True,
        max_steps=args.max_steps,  # Replace with the number of epochs you want
        per_device_train_batch_size=args.batch_size,  # Replace with your training batch size
        gradient_accumulation_steps=args.gas,
        save_steps=args.save_steps,
    )

    # 6. Train the Model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
# trainer.save_model("./output")  # Replace with the path where you want to save the model

if __name__ == "__main__":
    parser = ArgumentParser(description='Pretraining Script')
    parser.add_argument('--hs', type=int, default=256, help='hidden size')
    parser.add_argument('--na', type=int, default=4, help='number of attention heads')
    parser.add_argument('--nl', type=int, default=4, help='number of layers')
    parser.add_argument('--ffn', type=int, default=1024, help='ffn size')
    parser.add_argument('--save_path', type=str, default='../models/test', help='path to save model')
    parser.add_argument('--max_steps', type=int, default=160000, help='number of training steps')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gas', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--save_steps', type=int, default=20000, help='save model every n steps')
    args = parser.parse_args()
    main(args)