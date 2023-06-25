from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)






device = "auto"
model_path = "outputs/merged_model"             # Path to the combined weights

# Prompt should be in this style due to how the data was created
prompt = "#### Human: What is the capital of Australia?#### Assistant:"





bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map=device, 
    # load_in_8bit=True,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer(prompt, return_tensors="pt")
if device != "cpu":
    inputs = inputs.to('cuda')
del inputs['token_type_ids']
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=60, max_new_tokens=100)
output = tokenizer.decode(output[0], skip_special_tokens=True)


print(output.split("#### Assistant: ")[1])