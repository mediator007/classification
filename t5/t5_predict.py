import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained('C:/Users/User/Desktop/Programming/classification/t5/rut5-model-3')
tokenizer = T5Tokenizer.from_pretrained('C:/Users/User/Desktop/Programming/classification/t5/rut5-model-3')

def answer(x, **kwargs):
    inputs = tokenizer(x, return_tensors='pt').to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(**inputs, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


print(answer('Клиент менеджер'))