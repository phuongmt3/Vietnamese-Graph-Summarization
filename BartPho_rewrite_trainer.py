import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
from utils import cal_rouge
from load_data import SummDataset

wandb.login(key="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

model_path="vinai/bartpho-syllable-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # vocab_size = tokenizer.vocab_size
    # preds = np.clip(preds, 0, vocab_size - 1)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # labels = np.clip(labels, 0, vocab_size - 1)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print('Pred:', decoded_preds[:5])
    print('Label:', decoded_labels[:5])
    result = cal_rouge(decoded_labels, decoded_preds)
    result = {'p': round(result[3]*100, 2), 'r': round(result[4]*100, 2), 'f1': round(result[5]*100, 2)}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
#     print(result)
    return result



def main():
    train_set = SummDataset('/content/drive/MyDrive/Summarization/data', mode='train', limit=200)
    val_set = SummDataset('/content/drive/MyDrive/Summarization/data', mode='val', limit=100)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.generation_config.max_new_tokens = 512
    model.generation_config.num_beams = 3
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  # , label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/content/drive/MyDrive/Summarization/checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        # optim="paged_adamw_8bit",
        weight_decay=1e-6,
        # warmup_ratio=1,
        # fp16=True,
        # fp16_full_eval=True,

        num_train_epochs=10,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to="wandb",
        run_name='Summ_Bart'
    )

    # early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3,
    #                                                early_stopping_threshold=0.001)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping_callback]
    )
    trainer.train(resume_from_checkpoint=False)
    wandb.finish()
