import datasets
import pandas as pd
import transformers
import matplotlib.pyplot as plt
import shap
import os
os.environ['QT_QPA_PLATFORM'] = 'wayland'

# load the emotion dataset
dataset = datasets.load_dataset("emotion", split="train")
data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "nateraw/bert-base-uncased-emotion", use_fast=True
)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "nateraw/bert-base-uncased-emotion"
).to("cpu")

# build a pipeline object to do predictions
pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device="cpu",
    return_all_scores=True,
)

explainer = shap.Explainer(pred)
shap_values = explainer(data["text"][:3].tolist())

# Create a force plot for the first instance
force_plot = shap.plots.force(explainer.expected_value[0], shap_values[0])

shap.save_html('shap_plot.html', force_plot)
print("SHAP plot saved to shap_plot.html")