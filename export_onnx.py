import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig, BertTokenizer


class CharEmbedding(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.bert = BertModel(self.bert_config)
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)
    def expand_char_tophones(self,expand_length,word_bert_vec,max_char_length=256):
    
        reps = expand_length.float()
        reps = (reps+0.5).int()
        dec_lens = reps.sum(dim=1)
        max_len = torch.max(dec_lens)
    
        pad_zeros = torch.zeros(reps.size(0),1).int()
        reps = torch.cat((pad_zeros,reps),axis=1)
    
        reps_cumsum = torch.cumsum(reps,dim=1)[:,None,:]
        reps_cumsum = reps_cumsum.float()
    
        range_ = torch.arange(max_char_length)[None, :,None]
        mult = ((reps_cumsum[:, :, :-1] <= range_) &
                    (reps_cumsum[:, :, 1:] > range_))
        mult = mult.float()
        phone_word_length  = torch.matmul(mult, word_bert_vec)
        return phone_word_length,dec_lens


    def forward(self, inputs_ids,length,max_char_length=1024):

        x_len = input_ids.shape[1]
        inputs_masks = torch.ones(x_len,dtype=torch.int64).reshape(1,-1)
        tokens_type_ids = torch.zeros(x_len,dtype=torch.int64).reshape(1,-1)

        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)
        char_embeds = self.proj(out_seq[0])
        fix_phone_embeds,real_lens = self.expand_char_tophones(length,char_embeds,max_char_length)

        return fix_phone_embeds


if __name__ == "__main__":
    device = torch.device("cpu")
    char_model = CharEmbedding(r'./bert')
    print ("first1")
    char_model.load_state_dict(
            torch.load(os.path.join('./bert', 'prosody_model.pt'),map_location="cpu"),strict=False
        )
    char_model.eval()
    char_model.to(device)
    text = ["[PAD]","您","好",",","我","是","顺","丰","智","能","客","服","。","[PAD]"]
    input_ids = [0, 2644, 1962, 117, 2769, 3221, 7556, 705, 3255, 5543, 2145, 3302, 511, 0]
    #length = [('[PAD]', 1), ('您', 2), ('好', 2), ('，', 1), ('我', 2), ('是', 2), ('顺', 2), ('丰', 2), ('智', 2), ('能', 2), ('客', 2), ('服', 2), ('。', 1), ('[PAD]', 1)]
    length = [1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,1, 1]
    print ("input_ids:{}".format(input_ids))
    input_ids = torch.LongTensor([input_ids]).to(device)
    length = torch.LongTensor([length]).to(device)
    print (length)

    torch.onnx.export(
                      char_model,
                      (input_ids,length),
                      'prosody.onnx',
                       opset_version=13,
                       input_names=[
                           "word_input_ids",
                           "expand_length",
                           ],
                       output_names=["bert_vec"],
                       dynamic_axes={
                           "word_input_ids":{1:"L"},
                           "expand_length":{1:"L"},
                           "bert_vec":{1:"Le"},
                           }
                        )
