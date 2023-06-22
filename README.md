# Wizardlm Alpaca Dolly Orca Open_LLaMa_13b
An Open_LLaMA-13B model trained on custom explain tuned datasets, created using Instructions and Input from WizardLM, Alpaca & Dolly-V2 datasets and applying Orca Research Paper dataset construction approaches.


# Dataset

We trained [OpenLLaMa-13B model](https://github.com/openlm-research/open_llama) on custom explain tuned [WizardLM ~70K](https://github.com/nlpxucan/WizardLM), [Alpaca dataset ~52K](https://crfm.stanford.edu/2023/03/13/alpaca.html)  & [Dolly-V2 ~15K](https://github.com/databrickslabs/dolly) created using approaches from [Orca Research Paper](https://arxiv.org/abs/2306.02707). 

We leverage all of the 15 system instructions provided in Orca Research Paper. to generate custom datasets, in contrast to vanilla instruction tuning approaches used by original datasets.

This helps student model aka [wizardlm_alpaca_dolly_orca_open_llama_13b](https://huggingface.co/psmathur/wizardlm_alpaca_dolly_orca_open_llama_13b) to learn ***thought*** process from teacher model, which is ChatGPT (gpt-3.5-turbo-0301 version).

Please see below example usage how the **System** prompt is added before each *instruction*.

# Training

The training configurations are provided in the table below.

The training takes on 8x A100(80G) GPUs and lasts for around 15 Hours for cost of $180 using [Lambda Labs](https://lambdalabs.com)

We used DeepSpeed with Zero-3 approaches for parallel gpu training by writing our own fine tunning scripts plus leveraging some of the model training code provided by amazing [OpenAlpaca repo](https://github.com/yxuansu/OpenAlpaca)

Here are some of params used during training:

|||
|:-------------:|:-------------:|
|*batch_size*|16|
|*train_micro_batch_size_per_gpu*|2|
|*gradient_accumulation_steps*|1|
|*Learning rate*|2e-5|
|*Max length*|1024|
|*Epochs*|3|



# Example Usage

Below shows an example on how to use this model

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Hugging Face model_path
model_path = 'psmathur/wizardlm_alpaca_dolly_orca_open_llama_13b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)


#generate text function
def generate_text(system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n#\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n#\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    print(f'[!] Response: {string}')

# same prompt as provided by Orca Research Paper
system = 'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.'
instruction = 'Use the given data to calculate the median.'
input = '[5,2,3,4,1]'
generate_text(system, instruction, input)

```

**P.S. I am #opentowork and #collaboration, if you can help, please reach out to me at psmathur.public@gmail.com**

Next Goals:
1) Try more data like actually using FLAN-v2, just like Orka Research Paper (I am open for suggestions)
2) Try smaller OpenLLaMA models 7B and 3B
3) Provide more options for Text generation UI. (may be https://github.com/oobabooga/text-generation-webui)
4) Provide 4bit GGML/GPTQ quantized model (may be [TheBloke](https://huggingface.co/TheBloke) can help here)


Reference:
If you found wizardlm_alpaca_dolly_orca_open_llama_13b useful in your research or applications, please kindly cite using the following BibTeX:

```
@misc{wizardlm_alpaca_dolly_orca_open_llama_13b,
  author = {Pankaj Mathur},
  title = {wizardlm_alpaca_dolly_orca_open_llama_13b: An explain tuned OpenLLaMA-13b model on custom wizardlm, alpaca, & dolly datasets},
  year = {2023},
  publisher = {GitHub, HuggingFace},
  journal = {GitHub repository, HuggingFace repository},
  howpublished = {\url{https://github.com/pankajarm/wizardlm_alpaca_dolly_orca_open_llama_13b}, \url{https://https://huggingface.co/psmathur/wizardlm_alpaca_dolly_orca_open_llama_13b}},
}
```
```
@software{openlm2023openllama,
  author = {Xinyang Geng and Hao Liu},
  title = {OpenLLaMA: An Open Reproduction of LLaMA},
  month = May,
  year = 2023,
  url = {https://github.com/openlm-research/open_llama}
}
```
```
@misc{openalpaca,
  author = {Yixuan Su and Tian Lan and Deng Cai},
  title = {OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yxuansu/OpenAlpaca}},
}
```
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
