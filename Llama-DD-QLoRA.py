import re
import gc

from datasets import load_dataset
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

use_4bit = True
bnb_4bit_compute_dtype = "float16"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

def inference(query, model, tokenizer):
    query = f"<|start_header_id|>user A<|end_header_id|>\n<|bot_id|>{query}<|eot_id|>\n"
    inputs = tokenizer.encode(query, return_tensors='pt').to(device)
    response = model.generate(inputs, max_length=256, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(response[0], skip_special_tokens=False)
    return response

dataset = load_dataset("li2017dailydialog/daily_dialog")

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
special_tokens_dict = {'additional_special_tokens': ["<|start_header_id|>", "<|end_header_id|>", "<|bot_id|>", "<|eot_id|>"]}
tokenizer.add_special_tokens(special_tokens_dict)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

original_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-3B',
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config, 
    device_map=device 
)
original_model.resize_token_embeddings(len(tokenizer))

original_model.config.use_cache = False
original_model.config.pretraining_tp = 1

config = LoraConfig(
    r=8, #Rank
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    bias="none",
    lora_dropout=0.1,  # Conventional
    use_rslora = True,
    task_type=TaskType.CAUSAL_LM,
)

original_model.gradient_checkpointing_enable()

# fix gradient error when using model checkpointing
if hasattr(original_model, "enable_input_require_grads"):
        original_model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    original_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

model = get_peft_model(original_model, config)

def preprocess_function(examples):
    dialogues = [ex['dialog'] for ex in examples]
    act = [ex['act'] for ex in examples]
    emotion = [ex['emotion'] for ex in examples]
    
    inputs = []
    for dialog_turn in dialogues:
        prompt = ""
        for i, sentence in enumerate(dialog_turn):
            if i == 0 or i%2 == 0:
                prompt = prompt +  f"<|start_header_id|>user A<|end_header_id|>\n<|bot_id|>{sentence}<|eot_id|>\n"
            else:
                prompt = prompt +  f"<|start_header_id|>user B<|end_header_id|>\n<|bot_id|>{sentence}<|eot_id|>\n"
        
        prompt = prompt + f"{tokenizer.eos_token}"
        
        inputs.append(prompt)
        
    model_inputs = tokenizer(inputs, max_length=256, padding=True, truncation=True, return_tensors="pt")
    return model_inputs

train_dataloader = DataLoader(dataset['train'], batch_size=16, collate_fn=preprocess_function)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 100
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=True)

    print()
    print("Inference:")
    print(inference("Say , Jim , how about going for a few beers after dinner ?", model, tokenizer))
    print()

    for batch in progress_bar:
        # Move data to device
        # each element from the batch shape is [N, seq_length (not fixed)]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        
        progress_bar.set_postfix({"Loss": loss.item()})
        # Memory Optimization
        del batch
        del loss
        del input_ids
        del attention_mask
        del labels
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Training loss: {avg_train_loss:.4f}")

    model.save_pretrained("./fine-tuned-llama3.2-3B-DD-QLoRA3-v2.0")
    tokenizer.save_pretrained("./fine-tuned-llama3.2-3B-DD-QLoRA3-v2.0")