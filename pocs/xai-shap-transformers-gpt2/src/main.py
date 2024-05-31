from transformers import AutoModelForCausalLM, AutoTokenizer
import shap
from IPython.display import display

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("gpt2").cpu()

model.config.is_decoder = True
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0.7,
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}

s = ["I enjoy walking with my cute cats"]
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(s)
shap.plots.text(shap_values)