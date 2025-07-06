import torch
import torch.nn as nn
import torch.nn.functional as F
# Imported all modules required

class BahdanauAttention(nn.Module):
    def __init__(self,encoder_hidden_dim,decoder_hidden_dim,attention_dim):
        super(BahdanauAttention,self).__init__()
        # Layers for score function
        self.W_encoder = nn.Linear(encoder_hidden_dim,attention_dim)
        self.W_decoder = nn.Linear(decoder_hidden_dim,attention_dim)
        self.v = nn.Linear(attention_dim,1,bias=False)

    def forward(self,encoder_outputs,decoder_hidden,mask=None):
        # apply W_encoder(h_i) for all encoder inputs
        encoder_projection = self.W_encoder(encoder_outputs)
        # apply W_decoder(s_t) for all decoder inputs
        decoder_projection = self.W_decoder(decoder_hidden).unsqueeze(1)
        # add and apply tanh
        energy = torch.tanh(encoder_projection + decoder_projection)
        attention_scores = self.v(energy).squeeze(-1)
        # mask padding if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e10)
        attention_weights = F.softmax(attention_scores,dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1) , encoder_outputs).squeeze(1)

        return context , attention_weights
    



