import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained('./rut5-model')
tokenizer = T5Tokenizer.from_pretrained('./rut5-model')

def answer(x, **kwargs):
    inputs = tokenizer(x, return_tensors='pt').to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


print(answer('как заключить договор присоединения для услуг связи?'))