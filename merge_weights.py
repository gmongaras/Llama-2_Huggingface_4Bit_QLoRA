import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import shutil



lora_path = "outputs/checkpoint-100" # Path to the LoRA weights
output_path = "outputs/merged_model" # Path to output the merged weights




peft_model_id = lora_path
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

shutil.copytree(peft_config.base_model_name_or_path, output_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pt', "*.pth", "*.bin"))

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    sub_mod = model.get_submodule(key)
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    if isinstance(sub_mod, peft.tuners.lora.Linear):
        sub_mod.merge()
        bias = sub_mod.bias is not None
        new_module = torch.nn.Linear(sub_mod.in_features, sub_mod.out_features, bias=bias)
        new_module.weight.data = sub_mod.weight
        if bias:
            new_module.bias.data = sub_mod.bias
        model.base_model._replace_module(parent, target_name, new_module, sub_mod)

model = model.base_model.model

# Save the model
model.save_pretrained(output_path)