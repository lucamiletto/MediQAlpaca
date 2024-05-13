# 🦙🩺 MediQAlpaca

MediQAlpaca is a specialized Large Language Model (LLM) fine-tuned from the Alpaca model for biomedical question-answering tasks. It was used for assessesing the impact of domain-specific fine-tuning on LLM performance using the PubMedQA dataset and investigates how this enhanced model can support healthcare professionals in their roles. The findings from the creation of the model suggest that domain-specific fine-tuning not only boosts the model's applicability in specialized areas but also enhances medical professionals' decision-making efficiency, information retrieval and educational efforts. Future research should seek to broaden the model's functionality to accommodate multilingual and multimodal inputs, thereby improving its global utility in healthcare. Efforts should also be made to integrate the model seamlessly into clinical workflows and to develop an automated update mechanism. Additional recommendations delve into Explainable AI, the experimentation with larger models, and the performance of cost-effectiveness analyses to ensure these AI tools are economically viable, supporting their practical application in healthcare environments and enhancing operational efficiency.

This repository contains code that was used to create MediQAlpaca. The model was finetuned on Colab using a V100 GPU with the whole process taking 6:33:08h and costing less than $4.


### Local Setup

Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preprocessing

Preprocessing of the PubMedQA dataset portion is done using (`preprocessing.py`).
PubMedQA is made to train classificaiton into yes/no/maybe, input and answers and context had to be extracted.


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


### Inference and Evaluation

Inference was done using the (`inference.jpynb`). 
The dedicated script aided in aswering the (`test_data_s.json`) test set and processed the results in a manner such that metrics and graphical evaluation of the model results could be conduted.
Evaluation is performed in (`evaluation.jpynb`) and MediQAlpaca is compared against the base model.

##### ROUGE Scores for Native Alpaca

| Scores    | Precision | Recall  | F-score |
|-----------|-----------|---------|---------|
| ROUGE 1   | 0.2470    | 0.2401  | 0.2146  |
| ROUGE 2   | 0.0884    | 0.0808  | 0.0735  |
| ROUGE L   | 0.1798    | 0.1704  | 0.1533  |
| ROUGE LSum| 0.1800    | 0.1704  | 0.1533  |

##### ROUGE Scores for MediQAlpaca

| Scores    | Precision | Recall  | F-score |
|-----------|-----------|---------|---------|
| ROUGE 1   | 0.4601    | 0.2855  | 0.3273  |
| ROUGE 2   | 0.1791    | 0.1084  | 0.1253  |
| ROUGE L   | 0.3437    | 0.2141  | 0.2448  |
| ROUGE LSum| 0.3436    | 0.2139  | 0.2446  |

##### METEOR and BLEU Scores

| Score  | Alpaca-Native | MediQAlpaca |
|--------|---------------|-------------|
| METEOR | 0.1846        | **0.2384**  |
| BLEU   | 0.0476        | **0.059**   |



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


### Notes

- We can likely improve our model performance significantly if we had a larger dataset.
- Work done here should be takes as a baseline for understanding costs and complexities of finetuning LLMs when significative resource costraints exist

### Resources

- [alpaca-lora](https://github.com/tloen/alpaca-lora), code for finetuning llama based models using LoRA
- [Colab_For_Alpca_Lora](https://github.com/TianyiPeng/Colab_for_Alpaca_Lora), code for finetuning llama based models using colab
- [PubMedQA](https://github.com/pubmedqa/pubmedqa)
- [alpaca-native](<https://huggingface.co/chavinlo/alpaca-native>) a replication using the original Alpaca code


