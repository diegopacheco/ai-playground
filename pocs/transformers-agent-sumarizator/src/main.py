from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizationAgent:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt')
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

if __name__ == '__main__':
    agent = TextSummarizationAgent()
    text = "The history of cheesemaking dates back to 8000 BC. It was first discovered by the nomadic tribes of the Middle East who stored milk in vessels made from the stomachs of animals. The natural enzymes in the stomach would cause the milk to separate into curds and whey. The curds could then be seasoned with herbs and eaten."
    print(agent.summarize(text))
