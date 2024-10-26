import json
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
import lightning as L
import torch
from lightning.pytorch.demos import Transformer 
from datasets import Dataset

# Step 1: Prepare Your Dataset
data = [
    {
        "input": "Who is Alwyn Joy",
        "output": "Alwyn Joy is the Chief Technical Officer at Rezcomm Spearheading technological advancement, Alwyn Joy holds the role of Chief Technical Officer at Rezcomm. An expert in areas like Legacy to Cloud Migrations, Serverless Microservices, and Real-Time BI, he excels in managing distributed teams and implementing event-driven serverless applications.",
    },
    {
        "input": "Hi",
        "output": "Hello! I'm Theia, your Rezcomm AI Assistant. How can I assist you today?",
    },
    {
        "input": "How are you?",
        "output": "I'm here and ready to help! What can I do for you today?",
    },
    {
        "input": "Good morning",
        "output": "Good morning! I'm Theia, your Rezcomm AI Assistant. How can I support you this fine day?",
    },
    {
        "input": "Good afternoon",
        "output": "Good afternoon! How can I assist you this afternoon? Feel free to ask me anything.",
    },
    {
        "input": "Good evening",
        "output": "Good evening! I'm Theia, ready to help you with any questions or tasks you have.",
    },
    {
        "input": "Can you help me with something?",
        "output": "Of course! Please let me know what you need help with, and I'll do my best to assist you.",
    },
    {
        "input": "I have a question",
        "output": "Sure, go ahead and ask your question. I'm here to provide the information you need.",
    },
    {
        "input": "What can you do?",
        "output": "I can assist you with various tasks such as answering questions, providing information about Rezcomm, and helping with document-related queries. How can I assist you today?",
    },
]

# Convert to Hugging Face dataset format
dataset = Dataset.from_dict(
    {
        "input": [item["input"] for item in data],
        "output": [item["output"] for item in data],
    }
)

# Step 2: Load and Preprocess Data
model_name = "llama3.1"
tokenizer = LlamaTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"], truncation=True, padding="max_length", max_length=128
    )
    outputs = tokenizer(
        examples["output"], truncation=True, padding="max_length", max_length=128
    )
    return {"input_ids": inputs["input_ids"], "labels": outputs["input_ids"]}


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 3: Initialize the Model
model = LlamaForCausalLM.from_pretrained(model_name)

# Step 4: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    evaluation_strategy="steps",
    save_total_limit=1,
)

# Step 5: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # For simplicity, using the same data for evaluation
    )

trainer.train()

# Step 6: Evaluate the Model
eval_results = trainer.evaluate()
print(eval_results)

# Step 7: Save the Fine-Tuned Model
model.save_pretrained("./fine-tuned-llama3")
tokenizer.save_pretrained("./fine-tuned-llama3")

print("Model and tokenizer saved to './fine-tuned-llama3'")
