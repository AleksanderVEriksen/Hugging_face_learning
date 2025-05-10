import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*transformers.*")

from transformers import pipeline
import torch
# This code uses the Hugging Face Transformers library to create a text generation pipeline using the DistilGPT-2 model.
# It generates text based on a given prompt and prints the generated text.
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device= 0 if torch.cuda.is_available() else -1,  # Use GPU if available)
)
res = generator("Hello, I'm a text generator! This will be", max_length=50, truncation=True ,num_return_sequences=2)
print(res)

# This code uses the Hugging Face Transformers library to create a zero-shot classification pipeline using the BART model.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sequence = "I'm going to the gym to work on my bench press."
labels = ["fitness", "health", "sports"]
res = classifier(sequence, labels)
print(res)


# This code uses the Hugging Face Transformers library to create a sentiment analysis pipeline using the DistilBERT model.
from transformers import pipelines
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline('sentiment-analysis')

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
)
res = classifier("I love using Hugging Face Transformers!")
print(res)

# This code uses the Hugging Face Transformers library to create a tokenizer using the DistilBERT model.
sequence = "Using Hugging Face Transformers is great for NLP tasks."
res = tokenizer(sequence, return_tensors="pt")
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)