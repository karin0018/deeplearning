from torch import nn
from torch.nn import functional as F

# Define model
class LSTM(nn.Module):
    def __init__(self, vocab_size, pad_idx, embed_size = 200, batch_first=False,hidden_size = 128, num_layers=1,bidirectional=False, dropout=0.4,labels=2):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_size,padding_idx=pad_idx)
        self.encoder = nn.LSTM(input_size=embed_size,batch_first=batch_first, hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        
        self.decoder = nn.Linear(num_directions * hidden_size, labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        x = self.embedding(x)
        
        encoder_out, (_,_) = self.encoder(x) 
        """
        encoder_output shape : [seq_length, batch_size, num_directions * hidden_size]
        """
                
        decoder_out = self.decoder(encoder_out[:,-1,:])
        """
        we choice the last step of lstm output as the sentence representation.
        """
        output = self.dropout(decoder_out)
        """
        use the dropout for the full-connnected layer output
        """

        return output
        
        
        