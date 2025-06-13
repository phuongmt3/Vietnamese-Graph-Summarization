# Vietnamese Graph Summarization (Frozen Fusion)
Source code for The International Conference on Asian Language Processing (IALP) 2023 paper: Contrastive Hierarchical Discourse Graph for Vietnamese Extractive Multi-Document Summarization

## Data preprocessing
- All article raw texts and golden summaries need to be tokenized with the corresponding tokenizer
- Each sentence is passed through PhoBERT-base model to obtain embedding vector for every single words.
- Data in each cluster need to be transformed into a dictionary containing cluster information itself and a list of documents. Each document is a dictionary containing its information and a list of sentences. Each sentence is a dictionary including information such as its raw text, original paragraph_id it belongs to, and a list of spans. Each span represents a word in the sentence, which is also a dictionary with information of the word itself, word order in sentence, word vector.

## Train and Validation
Make use of data in train set and validation set. You can change the arguments and run file train.py.

## Test
Make use of data in test set. You can load your own trained model and run file test.py, which will create a 'VLSP Dataset/results.txt' file containing all created summaries.
You can print out the result on your own or upload the result file to [AI Hub](https://aihub.vn/competitions/341) for the detailed report.

## References

```
@INPROCEEDINGS{10337087,
  author={Mai, Tu-Phuong and Nguyen, Quoc-An and Can, Duy-Cat and Le, Hoang-Quynh},
  booktitle={2023 International Conference on Asian Language Processing (IALP)}, 
  title={Contrastive Hierarchical Discourse Graph for Vietnamese Extractive Multi-Document Summarization}, 
  year={2023},
  volume={},
  number={},
  pages={118-123},
  doi={10.1109/IALP61005.2023.10337087}}
```
