import sentencepiece as spm
from model import ModelArgs

data_path = ModelArgs.dataset_path
# Eğitim parametreleri
spm.SentencePieceTrainer.train(
    input=data_path,           
    model_prefix='tokenizer',       
    vocab_size=ModelArgs.vocab_size,               
    model_type='bpe',               
    character_coverage=1.0,         
    pad_id=0,                       
    unk_id=1,                       
    bos_id=2,                       
    eos_id=3,                       
    byte_fallback=True              
)

print(f'data path: {data_path}\n')
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

text = " Merhaba dünya!"
print("Tokenlar:", sp.encode(text, out_type=str))
print("ID'ler:", sp.encode(text, out_type=int))
