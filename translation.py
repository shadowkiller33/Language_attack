from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import pickle
#open a csv file and read it
import csv
harmful_prompts = []
with open("harmful_behaviors.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        harmful_prompts.append(row[0])
# with open('harmful_behaviors.csv', 'r') as f:
#     data = f.readlines()
harmful_prompts = harmful_prompts[1:201]

minor_languages = ['mri_Latn', 'khk_Cyrl', 'luo_Latn', 'kam_Latn', 'jav_Latn','ibo_Latn','hye_Armn','hau_Latn','urd_Arab']
high_languages = ['zho_Hans', 'rus_Cyrl', 'spa_Latn', 'por_Latn', 'fra_Latn','deu_Latn','ita_Latn','nld_Latn','tur_Latn']
middle_languages = ['xho_Latn', 'sot_Latn', 'slv_Latn', 'slk_Latn', 'mlt_Latn','lit_Latn','est_Latn','bos_Latn','afr_Latn']
tokenizer = AutoTokenizer.from_pretrained(
    "nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("nllb-200-1.3B", use_auth_token=True).cuda()


for minor_language in high_languages:
    translated_prompts = []
    for id, prompt in tqdm(enumerate(harmful_prompts)):

        inputs = tokenizer(prompt, return_tensors="pt")
        #move to cuda
        inputs = {k: v.cuda() for k, v in inputs.items()}
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[minor_language], max_length=100
        )
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        translated_prompts.append(translation)
        print(translation)
    with open(f'/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/high/{minor_language}.pkl', "wb") as f:
        pickle.dump(translated_prompts, f)
    # with open(f'/apdcephfs_cq2/share_1603164/data/lingfengshen/trl/{minor_language}.pkl', 'rb') as f:
    #     data = pickle.load(f)