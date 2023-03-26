# pip install transformers sentencepiece torch pandas openpyxl
import pandas as pd
from random import shuffle

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import trange
import random
import numpy as np

# raw_model = './rut5-base-multitask'
raw_model = 'C:/Users/User/Desktop/Programming/classification/t5/rut5-base' 
# use .cuda() if cuda install from
# https://developer.nvidia.com/cuda-downloads
# for training on GPU
model = T5ForConditionalGeneration.from_pretrained(raw_model).cuda()
tokenizer = T5Tokenizer.from_pretrained(raw_model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


batch_size = 16  # сколько примеров показываем модели за один шаг
report_steps = 200  # раз в сколько шагов печатаем результат
epochs = 8  # сколько раз мы покажем данные модели


data = pd.read_excel('data3.xlsx', usecols=['Questions', 'Answers']) # usecols - names of columns in xlsx
text = data['Questions'].tolist()
labels = data['Answers'].tolist()
pairs = [(text[i], labels[i]) for i in range(len(text))]
shuffle(pairs)

model.train()
losses = []
for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(pairs)
    for i in trange(0, int(len(pairs) / batch_size)):
        batch = pairs[i * batch_size: (i + 1) * batch_size]
        # кодируем вопрос и ответ 
        x = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True).to(model.device)
        y = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True).to(model.device)
        # -100 - специальное значение, позволяющее не учитывать токены
        y.input_ids[y.input_ids == 0] = -100
        # вычисляем функцию потерь
        loss = model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            labels=y.input_ids,
            decoder_attention_mask=y.attention_mask,
            return_dict=True
        ).loss
        # делаем шаг градиентного спуска
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # печатаем скользящее среднее значение функции потерь
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))



new_model_name = 'rut5-model-3'  # название папки для сохранения
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
