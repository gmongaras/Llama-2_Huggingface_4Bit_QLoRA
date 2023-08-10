# Llama-2_Huggingface_4Bit_QLoRA
A working example of a 4bit QLoRA Falcon/Llama2 model using huggingface

To start finetuning, edit  and run `main.py`

Once finetuning is complete, you should have checkpoints in `./outputs`. Before running inference, we can combine the LoRA weights with the original weights for faster inference and smaller GPU requirements during inference. To do this, run the `merge_weights.py` script with your paths.

Finally, you can run `generate.py` for example generation given the merged model.

# Requirements
The python requirements to run the script are located in requirements.txt

You should also download the Falcon weights of the 7B model here `https://huggingface.co/tiiuae/falcon-7b` and put the files in a directory `./tiiuae/falcon-7b` or download the Llama-2 weights here `https://huggingface.co/meta-llama/Llama-2-7b-hf` and put them in a directory named `./llama-2`

# Multiple GPUs
This script does not support multi-gpus on 4-bit finetuning. If I find a way to do this, I will update the script.

# GPU Requirements
- The base model takes about 6 GB of memory.
- Finetuning depends on the adapter size, batch size, max length, etc. In the
  current configuration, the memory usage is about 8GB.

# Issues
1. If there is a shape error upon training, then bitsandbytes and/or peft are having issues. The best way to get around this issue is to completely uninstall them and reinstall them from the source:
```
python -m pip uninstall bitsandbytes transformers accelerate peft -y
python -m pip install git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/accelerate.git git+https://github.com/timdettmers/bitsandbytes.git -U
```

2. If you get the error ` CUDA Setup failed despite GPU being available. Please run the following command to get more information`, then you need to build bitsandbytes from the source and put it in your bits and bytes site-package by following `https://github.com/oobabooga/text-generation-webui/issues/147`
