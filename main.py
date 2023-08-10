from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)



max_length = 128


# Model loading params
load_in_4bit = True

# LoRA Params
lora_alpha = 16             # How much to weigh LoRA params over pretrained params
lora_dropout = 0.1          # Dropout for LoRA weights to avoid overfitting
lora_r = 16                 # Bottleneck size between A and B matrix for LoRA params
lora_bias = "all"           # "all" or "none" for LoRA bias
model_type = "llama"     # falcon or llama
lora_target_modules = [     # Which modules to apply LoRA to (names of the modules in state_dict)
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
] if model_type == "falcon" else [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Trainer params
output_dir = "outputs"                              # Directory to save the model
optim_type = "adamw_8bit"                           # Optimizer type to train with 
learning_rate = 0.0005                              # Model learning rate
weight_decay = 0.002                                # Model weight decay
per_device_train_batch_size = 1                     # Train batch size on each GPU
per_device_eval_batch_size = 1                      # Eval batch size on each GPU
gradient_accumulation_steps = 16                    # Number of steps before updating model
warmup_steps = 5                                    # Number of warmup steps for learning rate
save_steps = 100                                    # Number of steps before saving model
logging_steps = 100                                 # Number of steps before logging








# Load in the model as a 4-bit or 8-bit model
if load_in_4bit == True:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b" if model_type == "falcon" else "meta-llama/Llama-2-7b-hf",
        trust_remote_code=True, 
        device_map="auto", 
        quantization_config=bnb_config
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b" if model_type == "falcon" else "meta-llama/Llama-2-7b-hf",
        trust_remote_code=True, 
        device_map="auto", 
        load_in_8bit=True,
    )



# Load in the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "tiiuae/falcon-7b" if model_type == "falcon" else "meta-llama/Llama-2-7b-hf",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token



# Load in the dataset and map using the tokenizer
dataset = load_dataset("squad")
"""
The dataset has context, questions, and answers.

For this example, I am just encoding the question and first answer.
when you would actually want the context and question.

We want the text string to be in the format
#### Human: {question}#### Assistant: {output}

We want to turn this into the format:
{
    "input_ids": input ids for the encoded instruction and input
    "labels": This is the input ids, but we put -100 where we want to mask the
                loss. We want to mask the loss for the instruction, input, and padding.
                We use -100 because PyTorch CrossEntropy ignores -100 labels.
    "attention_mask": attention mask so the model doesn't attend to padding
}
"""
def map_function(example):
    # Get the question and model output
    question = f"#### Human: {example['question'].strip()}"
    output = f"#### Assistant: {example['answers']['text'][0].strip()}"
    
    # Encode the question and output
    question_encoded = tokenizer(question)
    output_encoded = tokenizer(output, max_length=max_length-len(question_encoded["input_ids"]), truncation=True, padding="max_length")
    
    # Combine the input ids
    input_ids = question_encoded["input_ids"] + output_encoded["input_ids"]
    
    # The labels are the input ids, but we want to mask the loss for the context and padding
    labels = [-100]*len(question_encoded["input_ids"]) + [output_encoded["input_ids"][i] if output_encoded["attention_mask"][i] == 1 else -100 for i in range(len(output_encoded["attention_mask"]))]
    
    # Combine the attention masks. Attention masks are 0
    # where we want to mask and 1 where we want to attend.
    # We want to attend to both context and generated output
    attention_mask = [1]*len(question_encoded["input_ids"]) + output_encoded["attention_mask"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }
data_train = dataset["train"].map(map_function)
data_test = dataset["validation"].map(map_function)


# Adapt the model with LoRA weights
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias=lora_bias,
    task_type="CAUSAL_LM",
    inference_mode=False,
    target_modules=lora_target_modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    optim=optim_type,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    do_train=True,
    warmup_steps=warmup_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()