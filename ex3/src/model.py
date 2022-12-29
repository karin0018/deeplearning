from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer
import torch

# reference: https://zhuanlan.zhihu.com/p/535100411
# reference: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
# Define model
class Bert(nn.Module):
    def __init__(self, hidden_size = 768, labels=2, max_length=256, device='cuda'):
        super().__init__()
        # use simple bert
        print('prepare the pretrained model ... ')
        model_name = 'distilbert-base-uncased'
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, do_lower_case=True)
        # load pretrained model
        self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        self.decoder = nn.Linear(hidden_size, labels)
        
        self.max_length=max_length
        self.device = device
        
        print('prepare model have done!')
        
    def forward(self, batch_sentences): 
        """
        batch_sentences : [batch_size, seq]
        """
        sentence_tokenizer = self.tokenizer(batch_sentences, # sentence to encode
                                            truncation=True, # cut off the rest word if overlap max length
                                            padding = True,
                                            max_length = self.max_length,
                                            add_special_tokens = True
                                            )
        input_ids = torch.tensor(sentence_tokenizer['input_ids']).to(self.device)
        attention_mask = torch.tensor(sentence_tokenizer['attention_mask']).to(self.device)
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        """ 
        encoder_out <-> bert_out: tuple, includes
        last_hidden_state, pooler_output, hidden_states, attentions, cross_attentions, past_key_values
        """
        last_hidden_state = encoder_out[1] # [batch_size,  hidden_size]
        """
        I choice the pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) of each sentence as the representation.
        """
                
        output = self.decoder(last_hidden_state)

        return output
        
        
        