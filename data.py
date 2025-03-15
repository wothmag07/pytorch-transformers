import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class OpusBooks(Dataset):
    def __init__(self, dataset, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        sample = self.dataset[index]
        src_text, tgt_text = sample['translation'][self.src_lang], sample['translation'][self.tgt_lang]

        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        en_num_padding_tokens = self.seq_len - len(src_tokens) - 2
        de_num_padding_tokens = self.seq_len - len(tgt_tokens) - 1

        if en_num_padding_tokens < 0 or de_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        #Adding SOS and POS to the source sentences
        encoder_input = torch.cat([self.sos_token,
                                   torch.tensor(src_tokens, dtype=torch.long), self.eos_token,
                                   torch.tensor([self.pad_token] * en_num_padding_tokens, dtype=torch.long)], dim=0)
        # print(encoder_input.shape)

        #Adding SOS to the target sentences
        decoder_input = torch.cat([self.sos_token,
                                   torch.tensor(tgt_tokens, dtype=torch.long),
                                   torch.tensor([self.pad_token]* de_num_padding_tokens, dtype=torch.long)], dim=0)
        
        # print(decoder_input.shape)
        #Adding EOS to the predicted sentences
        target_op = torch.cat([torch.tensor(tgt_tokens, dtype=torch.long),
                               self.eos_token,
                               torch.tensor([self.pad_token]* de_num_padding_tokens, dtype=torch.long)], dim=0)
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert target_op.size(0) == self.seq_len

        """The attention_mask is used to ignore padding or unwanted tokens in attention computations,
           ensuring proper batching in training.The causal mask is specific to decoder-only and encoder-decoder models,
           preventing access to future tokens during training to avoid information leakage."""

        return {"encoder_input" : encoder_input, #(seq_len)
                "decoder_input" : decoder_input, #(seq_len)
                "targets" : target_op, #(seq_len)
                "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
                "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1, seq_len, seq_len)
                "src_txt" : src_text,
                "tgt_txt" : tgt_text} 


def causal_mask(size):
    mask_vector = torch.triu(torch.ones((1,size, size)), diagonal=1).type(torch.int)
    return mask_vector == 0

