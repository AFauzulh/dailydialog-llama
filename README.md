
# Open-domain Generative-based Multi-turn Chatbot using Large Language Model for Daily Conversation

- Model : Llama 3.2 3B
- Dataset : [DailyDialog Dataset](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- Fine Tuning:
    * QLoRA Adapter on Transformer Attentions

- UI:
    * Streamlit simple chat UI
 

## Fine-tuning

Note: \
If you already download the base pretrained model from huggingface you can create `model_cache` folder inside this repository place the base pretrained model into `model_cache` folder.

To manually fine-tune model, run the following script:

```bash
  python Llama-finance-QLoRA.py
```

## Deployment


**Pre-requisites :** \
    1. Download the [fine-tuned](https://drive.google.com/file/d/1J_xMjfMeiuAgD48jk1kWMY7A7bn1G7Mx/view?usp=sharing) models \
    2. Place the downloaded model into the repository then extract it 


To deploy this project run

### Step 1 : Build Docker Image
```bash
  docker build -t llama-dd-chatbot-app .
```

### Step 2 : Run Docker Container
```bash
  docker run -d --restart always --gpus all --name llama-dd-chatbot-app -p 8501:8501 llama-dd-chatbot-app
```

To stop and delete the container run
```bash
  docker stop llama-dd-chatbot-app && docker rm llama-dd-chatbot-app
```
