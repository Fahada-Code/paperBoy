"""QLoRA fine-tuning of the local relevance judge.

`dataset.py` is pure stdlib (turns the label store into an instruction dataset)
and safe to import anywhere. `train_judge.py` pulls the heavy ML stack (torch,
transformers, peft, trl) and is run as a script, so it is NOT imported here -
importing this package must stay cheap for the CLI.
"""
