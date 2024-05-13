# 🦙🩺 MediQAlpaca

This repository contains code for reproducing MediQAlpaca, a question answering generative LLM for healthcare professionals.

### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the Alpaca model,
as well as some code related to prompt construction and tokenization.

Example usage:

```bash
!python finetune.py \
    --base_model 'alpaca-native' \
    --data_path 'PubMedQA/10_percent_train_dataset.json' \
    --output_dir './lora-alpaca' \
```

We can also tweak our hyperparameters:

  ```bash
!python finetune.py \
    --base_model 'alpaca-native' \
    --data_path 'PubMedQA/10_percent_train_dataset.json' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 4500 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length;
   ```

### Inference

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

### Weights



### Notes

- We can likely improve our model performance significantly if we had a larger dataset.
- Work done here should be takes as a baseline for understanding costs and complexities of finetuning LLMs when significative resource costraints exist
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).

### Resources

- [alpaca-lora](https://github.com/tloen/alpaca-lora), code for finetuning llama based models using LoRA
- [Colab_For_Alpca_Lora](https://github.com/TianyiPeng/Colab_for_Alpaca_Lora), code for finetuning llama based models using colab
- [PubMedQA](https://github.com/pubmedqa/pubmedqa)
- [alpaca-native](<https://huggingface.co/chavinlo/alpaca-native>) a replication using the original Alpaca code


### Example outputs

**Instruction**: Can post-RT neck dissection be omitted for patients with head-and-neck cancer who have a negative PET scan after definitive radiation therapy?

**MediQAlpaca**: PET imaging after radiation is less sensitive, thus post-RT neck dissection can be omitted.

**alpaca-native**: No, post-RT neck dissection cannot be omitted.

**PubMedQA**: Positive PET post-RT indicates residual disease; negative PET indicated absence in only 14\%. Further research needed.

---

**Instruction**: Does stenting of left-sided colorectal cancer as a "bridge to surgery" adversely affect oncological outcomes?

**MediQAlpaca**: Does not affect long-term survival.

**alpaca-native**: No adverse effects on oncological outcomes.

**PubMedQA**: SEMS as a "bridge to surgery" has comparable survival to elective surgery. Confirmation needed in larger studies.

---

**Instruction**: Pulmonary atresia with VSD and multifocal pulmonary blood supply: does an intensive interventional approach improve the outcome?

**MediQAlpaca**: Intensive transcatheter approach improves pulmonary vascular growth and decreases Nakata index.

**alpaca-native**: No specific answer provided.

**PubMedQA**: Surgical and catheter-based approaches improve outcomes, require risk stratification and exit angiography.
