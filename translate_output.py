import json
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
with open('output/low/merged_output.json', 'r') as f:
    data = json.load(f)

sentences = []
cnt = 0
for i,x in enumerate(data):
    # print(i)
    if 'generated' in x.keys():
        if x['generated'] == '':
            cnt += 1
        sentences.append(x["generated"])
print(cnt)
lang_dict = {
    '0': 'hau_Latn',
    '1': 'hye_Armn',
    '2': 'ibo_Latn',
    '3': 'jav_Latn',
    '4': 'kam_Latn',
    '5': 'khk_Cyrl',
    '6': 'luo_Latn',
    '7': 'mri_Latn',
    '8': 'urd_Arab',
}



model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B").cuda()


# for minor_language in minor_languages:
translated_outputs = []
for id, prompt in tqdm(enumerate(sentences)):
    if prompt == '':
        translated_outputs.append({'content':'None'})
        continue
    #move to cuda
    index = int(id/200)
    minor_language = lang_dict[str(index)]
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-1.3B", src_lang=minor_language
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.cuda() for k, v in inputs.items()}
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'], max_length=100
    )
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f'The "{id}": {translation}')
    translated_outputs.append({'content':translation})

with open(f'./output/translated_output.json', "w") as f:
    json.dump(translated_outputs, f, indent=4)