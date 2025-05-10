from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*transformers.*")


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
)

X_train = ["I love using Hugging Face Transformers!", "This is a great library for NLP tasks.", "I am not a fan of this product."]

# Get the model's predictions
res = classifier(X_train)
print(res)

# Create a batch of inputs
batch = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
print(batch)

# Train the model
with torch.no_grad():
    outputs = model(**batch) # Dictionary of tensors
    logits = outputs.logits
    print(logits)
    predictions = F.softmax(logits, dim=-1)
    print(predictions)
    predicted_labels = torch.argmax(predictions, dim=-1)
    print(predicted_labels)