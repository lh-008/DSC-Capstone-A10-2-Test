from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer #transformer reinforcement learning  for ppo

from trl import DPOTrainer

model_name = "gpt2" #causal LM  
model_tokenizer = AutoTokenizer.from_pretrained(model_name)
model_tokenizer.pad_token = model_tokenizer.eos_token #pad_token is None by default

model = AutoModelForCausalLM.from_pretrained(model_name) # using pre-trained for now to test DPO implementation
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)  # is a static copy of the starting model so that model doesnt go to far away


##PPO Trainer setup
'''
config = PPOConfig( #setup the kubernetes wandb sweep later
    batch_size=32,
    mini_batch_size=8,
    ppo_epochs=8,
    learning_rate=1e-5,
    log_with=None,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=speaker,
    ref_model=speaker_ref,
    tokenizer=speaker_tokenizer,
)

device = ppo_trainer.accelerator.device
'''

#DPO Training setup for proof of concept
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dpo-speaker")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO preference strength")
    return parser.parse_args()

def trainDPOModel():

    training_args = TrainingArguments(
        output_dir="./dpo-speaker",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=AutoModelForCausalLM.from_pretrained(model_name), 
        args=training_args,
        train_dataset=preference_dataset,
        tokenizer=model_tokenizer,
        beta=0.1,
        max_length=512,
        max_prompt_length=384,  
    )

    dpo_trainer.train()

if __name__ == "__main__":
    args = parse_arguments()