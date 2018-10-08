import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttentionGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, fc_hidden, fc_output, da, r, num_layers=3, bidirec=False):
        """
        model idea is from paper: https://arxiv.org/pdf/1703.03130.pdf
        vocab: input_size = vocab_size
        embedding: embedding_size
        hidden: hidden_size
        fc_hidden: hidden_size (fully-connected)
        fc_output: output_size (fully-connected)
        da: attenion_dimension (hyperparameter)
        r: keywords (different parts to be extracted from the sentence)
        """
        super(SelfAttentionGRU, self).__init__()
        self.r = r
        self.da = da
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirec else 1

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True, 
                          bidirectional=bidirec)
        self.attn = nn.Linear(self.num_directions * hidden_size, 
                              self.da, 
                              bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.attn2 = nn.Linear(self.da, self.r, bias=False)
        self.attn_dist = nn.Softmax(dim=2)

        self.fc = nn.Sequential(
            nn.Linear(r * hidden_size * self.num_directions, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_output),
        )

    def init_GRU(self, batch_size, device):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return hidden.to(device)

    def penalization_term(self, A, device):
        """
        A : B, r, T
        Frobenius Norm
        """
        eye = torch.eye(A.size(1)).expand(A.size(0), self.r, self.r).to(device)  # B, r, r
        P = torch.bmm(A, A.transpose(1, 2)) - eye  # B, r, r
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5  # B, 1
        loss_P = torch.sum(loss_P) / A.size(0)  # 1
        return loss_P

    def forward(self, inputs, inputs_lengths, device=None):
        """
        inputs: B, T
         - B: batch_size
         - T: max_len = seq_len
        inputs_lengths: length of each sentences
        """
        embed = self.embed(inputs)  # B, T, V  --> B, T, E
        hidden = self.init_GRU(inputs.size(0), device)  # num_layers * num_directions, B, H

        # pack padded sequences
        packed = pack_padded_sequence(embed, inputs_lengths.tolist(), batch_first=True)
        # packed: B * T, E
        output, hidden = self.gru(packed, hidden)
        # output: B * T, num_directions * D
        # hidden: num_layers * num_direc, B, u

        # unpack padded sequences
        output, output_lengths = pad_packed_sequence(output, batch_first=True)
        # output: B, T, num_direc*u

        # Self Attention
        a1 = self.attn(output)  # Ws1(B, da, num_direc*u) * output(B, n, 2H) -> B, n, da
        tanh_a1 = self.tanh(a1)  # B, n, da
        score = self.attn2(tanh_a1)  # Ws2(B, r, da) * tanh_a1(B, T, da) -> B, n, r
        self.A = self.attn_dist(score.transpose(1, 2))  # B, r, T
        self.M = self.A.bmm(output)  # B, r, T * B, T, num_direc*u -> B, r, num_direc*u

        # Penalization Term
        loss_P = self.penalization_term(self.A, device=device)

        # Fully-Connected Layers
        # B, r, num_direc*u -> resize to B, r*num_direc*u -> B, H_f -> Relu -> B, 1
        output = self.fc(self.M.view(self.M.size(0), -1))  

        return output, loss_P

    def predict(self, inputs, inputs_lengths, device, thres=0.5):
        scores, _ = self.forward(inputs, inputs_lengths, device)
        return torch.sigmoid(scores).ge(thres).long()