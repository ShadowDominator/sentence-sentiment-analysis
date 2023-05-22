import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_sentence_analysis = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model_sentence_analysis = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
paragraph = """
I woke up this morning feeling refreshed and excited for the day ahead. 
I had a great night's sleep, and I was looking forward to spending time with my family and friends. 
I went for a walk in the park, and I enjoyed the beautiful weather. I also stopped by my favorite coffee shop and got a delicious cup of coffee. 
I felt so happy and content, and I knew that it was going to be a great day.

"""
def sentence_sentiment_model(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        result = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = result.logits.detach()
        probs = torch.softmax(logits, dim=1)
    pos_prob = probs[0][2].item()
    neu_prob = probs[0][1].item()  
    neg_prob = probs[0][0].item() 
    return {'Positive': [round(float(pos_prob), 2)],"Neutural":[round(float(neu_prob), 2)], 'Negative': [round(float(neg_prob), 2)]}

def sentence_sentiment(text):
    result = sentence_sentiment_model(text,tokenizer_sentence_analysis,model_sentence_analysis)
    return result

with gr.Blocks(title="Sentence",css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Sentence sentiment")
            with gr.Row():           
                with gr.Column():
                    inputs = gr.TextArea(label="sentence",value=paragraph,interactive=True)
                    btn = gr.Button(value="RUN")
                with gr.Column():
                    output = gr.Label(label="output")
                btn.click(fn=sentence_sentiment,inputs=[inputs],outputs=[output])
demo.launch()