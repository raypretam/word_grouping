## 🤗 Model on the Hub

The Hindi-Sanskrit translation model finetuned using contrastive loss on grouped Hindi data 
  👉 [Pretam/hindi_sanskrit](https://huggingface.co/Pretam/hindi_sanskrit)

# How to Get Started with the Model

Use the code below to get started with the model.

```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Pretam/hindi_sanskrit")
model = AutoModelForSeq2SeqLM.from_pretrained("Pretam/hindi_sanskrit")

article = "इसके लिए साधनों अनुष्ठान तो करना ही चाहिए।"
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("san_Deva"), # "san_Deva" languages-tag is required for the model to output Sanskrit.
    max_length=30
)

translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(translation)
```
