from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).to("cpu")

prompt = '''
    public class Calculator {
       public int add(int a, int b) {
            return a + b;
        }
    }
  
    public void testAddMethod(){
        <FILL_ME>
    }
'''

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cpu")
output = model.generate(
    input_ids,
    max_new_tokens=200,
)
output = output[0].to("cpu")

filling = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
print(prompt.replace("<FILL_ME>", filling))