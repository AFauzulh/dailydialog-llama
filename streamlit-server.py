import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModelForCausalLM
from huggingface_hub import login

login(token="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")

if torch.cuda.is_available():       
    device = torch.device("cuda")
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if "tokenizer" not in st.session_state.keys():
    print("Tokenizer not found in session state keys, loading tokenizer. . .")
    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained('./fine-tuned-llama3.2-3B-DD-QLoRA3-v2.0')
    print("Tokenizer has been successfully Loaded...")

if "model" not in st.session_state.keys():
    print("Model not found in session state keys, loading pretrained model. . .")

    original_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-3B',
        cache_dir="./model_cache",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config, 
        device_map=device
    )
    original_model.resize_token_embeddings(len(st.session_state["tokenizer"]))

    st.session_state["model"] = PeftModelForCausalLM.from_pretrained(
        model= original_model,
        model_id='./fine-tuned-llama3.2-3B-DD-QLoRA3-v2.0',
        peft_config=bnb_config,
        device_map=device,
    )

def inference_qlora_prompt(prompt, turn, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    response = model.generate(inputs, max_length=4096, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(response[0], skip_special_tokens=True)

    if st.session_state["turn"] == 0:
        response = response.split("\n")[3]
    else:
        response = response.split("\n")[3 + (turn*4)]

    return response

st.title("💬 DailyDialog Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello there !"}]

if "prompt" not in st.session_state:
    st.session_state["prompt"] = f"<|begin_of_text|>"

if "turn" not in st.session_state:
    st.session_state["turn"] = 0

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_message := st.chat_input():
    
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.chat_message("user").write(user_message)
    
    user_prompt = f"<|start_header_id|>user A<|end_header_id|>\n<|bot_id|>{user_message}<|eot_id|>\n"
    st.session_state["prompt"] = st.session_state["prompt"] + f"{user_prompt}"

    with st.spinner("generating response. . ."):
        response = inference_qlora_prompt(st.session_state["prompt"], st.session_state["turn"], st.session_state["model"], st.session_state["tokenizer"])
        bot_prompt = f"<|start_header_id|>user B<|end_header_id|>\n<|bot_id|>{response}<|eot_id|>\n"

        st.session_state["prompt"] = st.session_state["prompt"] + f"{bot_prompt}"
        st.session_state["turn"] = st.session_state["turn"] + 1

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)