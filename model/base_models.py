import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, padding_idx=0, dropout=0.3, freeze_embeddings=False):
        super(VanillaRNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, lengths, use_attention=False):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = hidden[-1]
        if use_attention:
            return output, last_hidden
        else:
            return self.fc(self.dropout(last_hidden))


class VanillaLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, padding_idx=0, dropout=0.3, freeze_embeddings=False):
        super(VanillaLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, lengths, use_attention=False):
        embedded = self.embedding(input_ids)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embeddings)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        last_hidden = hidden[-1]
        if use_attention:
            return output, last_hidden
        else:
            return self.fc(self.dropout(last_hidden))


class BidirectionalRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, padding_idx=0, dropout=0.3, freeze_embeddings=False):
        super(BidirectionalRNN, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx)
        self.bidirectionalrnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, lengths, use_attention=False):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.bidirectionalrnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        if use_attention:
            return output, hidden_cat
        else:
            return self.fc(self.dropout(hidden_cat))


class BidirectionalLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, padding_idx=0, dropout=0.3, freeze_embeddings=False):
        super(BidirectionalLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx)
        self.bidirectionallstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, lengths, use_attention=False):
        embedded = self.embedding(input_ids)
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.bidirectionallstm(packed_embedding)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        if use_attention:
            return output, hidden_cat
        else:
            return self.fc(self.dropout(hidden_cat))
        


class AttentionClassifier(nn.Module):
    def __init__(self,base_model,attention_module,hidden_dim,output_dim,dropout=0.3):
        super(AttentionClassifier,self).__init__()
        self.base_model = base_model
        self.attention = attention_module
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim,output_dim)


    def forward(self,input_ids,lengths):
        encoder_outputs,final_hidden = self.base_model(input_ids,lengths,use_attention=True)
        context_vector,attention_weights = self.attention(encoder_outputs,final_hidden)
        logits=self.fc(self.dropout(context_vector))
        return logits,attention_weights
    



