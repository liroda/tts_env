# 变长

def expand_char_to_phones(expand_length,word_bert_vec,max_char_length=256):

  reps = expand_length.float()
  reps = (reps + 0.5).int()
  dec_lens = reps.sum(dim=1)
  max_len = torch.max(dec_lens)

  pad_zeros = torch.zeros(reps.size(0),1).int()
  reps = torch.cat((pad_zeros,reps), axis=1)

  reps_cumsum = torch.cumsum(reps,dim=1)[:,None,:]
  reps_cumsum = reps_cumsum.float()

  range_ = torch.arange(max_char_length)[None,:,None]
  mult = (( reps_cumsum[:,:,:,:-1] <= range_ ) &
          ( reps_cumsum[:,:,1:] > range_))
  mult = mult.float()
  phone_word_length = torch.matmul(mult,word_bert_vec)
  return phone_word_length,dec_lens
