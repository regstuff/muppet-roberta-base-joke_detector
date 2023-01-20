# muppet-roberta-base-joke_detector
Detect "narrative-style" jokes, stories and anecdotes, spoken during speeches or conversations etc.

### What is this?
This model has been developed to detect "narrative-style" jokes, stories and anecdotes (i.e. they are narrated as a story) spoken during speeches or conversations etc. It works best when jokes/anecdotes are at least 40 words or longer. It is based on Facebook's [RoBerta-MUPPET](https://huggingface.co/facebook/muppet-roberta-base). 

The training dataset was a private collection of around 2000 jokes. This model has not been trained or tested on one-liners, puns or Reddit-style language-manipulation jokes such as knock-knock, Q&A jokes etc.

An example of a joke this model would detect: A nervous passenger is about to book a flight ticket, and he asks the airlines' ticket seller, 'I hope your planes are safe. Do they have a good track record for safety?' The airline agent replies, 'Sir, I can guarantee you, we've never had a plane that has crashed more than once.'

The model is available on [Hugging Face](https://huggingface.co/Reggie/muppet-roberta-base-joke_detector).
    
### Install these first
You'll need to pip install transformers & maybe sentencepiece

### How to use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, time
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = 'Reggie/muppet-roberta-base-joke_detector'
max_seq_len = 510

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_len)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

premise = """A nervous passenger is about to book a flight ticket, and he asks the airlines' ticket seller, "I hope your planes are safe. Do they have a good track record for safety?" The airline agent replies, "Sir, I can guarantee you, we've never had a plane that has crashed more than once." """
hypothesis = ""

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
is_joke = True if prediction[0] < prediction[1] else False

print(is_joke)
```
