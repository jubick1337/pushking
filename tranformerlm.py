#
#
# class PositionalEncoding(torch.nn.Module):
#     """
#     Pytorch module that creates a positional embedding with the same dimensions as the token embeddings.
#
#     Input dimension is: (batch_size, sequence_length, embedding_dimension)
#     Output dimension is: (batch_size, sequence_length, embedding_dimension)
#     """
#
#     def __init__(self, dim, max_sequence_length, device):
#         super().__init__()
#         self.dim = dim
#         self.max_sequence_length = max_sequence_length
#
#         positional_encoding = np.zeros((self.max_sequence_length, self.dim))
#         for pos in range(self.max_sequence_length):
#             for i in range(0, self.dim, 2):
#                 positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.dim)))
#                 positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.dim)))
#         self.positional_encoding = torch.from_numpy(positional_encoding).float().to(device)
#
#     def forward(self, x):
#         """
#         Adds the positional encoding to the token embeddings.
#         """
#
#         return x + self.positional_encoding[: x.size(1), :].to(x.device)
#
#
# class MaskedSelfAttention(torch.nn.Module):
#     """
#     Pytorch module for a self attention layer.
#     This layer is used in the MultiHeadedSelfAttention module.
#
#     Input dimension is: (batch_size, sequence_length, embedding_dimension)
#     Output dimension is: (batch_size, sequence_length, head_dimension)
#     """
#
#     def __init__(self, d_model, dim):
#         super().__init__()
#         self.d_model = d_model
#         self.dim = dim
#         self.queries = torch.nn.Linear(d_model, self.dim)
#         self.keys = torch.nn.Linear(d_model, self.dim)
#         self.values = torch.nn.Linear(d_model, self.dim)
#
#     def forward(self, x, mask):
#         """
#         Compute the self attention.
#
#         x dimension is: (batch_size, sequence_length, embedding_dimension)
#         output dimension is: (batch_size, sequence_length, dim)
#         mask dimension is: (batch_size, sequence_length)
#
#         mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
#         """
#
#         # x dimensions are: (batch_size, sequence_length, embedding_dimension)
#         # query, key, value dimensions are: (batch_size, sequence_length, dim)
#         query = self.queries(x)
#         key = self.keys(x)
#         value = self.values(x)
#
#         # Calculate the attention weights.
#         # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
#         attention_weights = torch.matmul(query, key.transpose(-2, -1))
#
#         # Scale the attention weights.
#         attention_weights = attention_weights / np.sqrt(self.dim)
#
#         # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
#         # This will make the softmax output 0 for these values.
#         mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
#         attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
#
#         # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
#         # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
#         attention_scores = attention_weights.softmax(-1)
#
#         # The attention scores are multiplied by the value
#         # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
#         # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
#         # Output dimensions are: (batch_size, sequence_length, head_dimension)
#         return torch.bmm(attention_scores, value)
#
#
# class MaskedMultiHeadedSelfAttention(torch.nn.Module):
#     """
#     Pytorch module for a multi head attention layer.
#
#     Input dimension is: (batch_size, sequence_length, embedding_dimension)
#     Output dimension is: (batch_size, sequence_length, embedding_dimension)
#     """
#
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         self.d_model = d_model
#         self.dim = d_model // num_heads
#         self.num_heads = num_heads
#
#         # Create the self attention modules
#         self.self_attentions = torch.nn.ModuleList([MaskedSelfAttention(d_model, self.dim) for _ in range(num_heads)])
#
#         # Create a linear layer to combine the outputs of the self attention modules
#         self.output_layer = torch.nn.Linear(num_heads * self.dim, d_model)
#
#     def forward(self, x, mask):
#         """
#         Compute the multi head attention.
#
#         x dimensions are: (batch_size, sequence_length, d_model)
#         mask dimensions are: (batch_size, sequence_length)
#         mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
#         """
#         # Compute the self attention for each head
#         # self_attention_outputs dimensions are:
#         # (num_heads, batch_size, sequence_length, head_dimension)
#         self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]
#
#         # Concatenate the self attention outputs
#         # self_attention_outputs_concatenated dimensions are:
#         # (batch_size, sequence_length, num_heads * head_dimension)
#         concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)
#
#         # Apply the output layer to the concatenated self attention outputs
#         # output dimensions are: (batch_size, sequence_length, embedding_dimension)
#         return self.output_layer(concatenated_self_attention_outputs)
#
#
# class FeedForward(torch.nn.Module):
#     """
#     Pytorch module for a feed forward layer.
#
#     A feed forward layer is a fully connected layer with a ReLU activation function in between.
#     """
#
#     def __init__(self, d_model, d_ffn):
#         super().__init__()
#         self.d_model = d_model
#         self.d_ffn = d_ffn
#         self.linear_expand = torch.nn.Linear(d_model, d_ffn)
#         self.linear_shrink = torch.nn.Linear(d_ffn, d_model)
#
#     def forward(self, x):
#         """
#         Compute the feed forward layer.
#         """
#         return self.linear_shrink(torch.relu(self.linear_expand(x)))
#
#
# class DecoderLayer(torch.nn.Module):
#     """
#     Pytorch module for an encoder layer.
#
#     An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.
#
#     Input dimension is: (batch_size, sequence_length, embedding_dimension)
#     Output dimension is: (batch_size, sequence_length, embedding_dimension)
#     """
#
#     def __init__(self, d_model, num_heads, d_hid, dropout):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_hid = d_hid
#         # self.dropout = dropout
#
#         self.mhsa = MaskedMultiHeadedSelfAttention(d_model, num_heads)
#         self.ff = FeedForward(d_model, d_hid)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.ln_1 = torch.nn.LayerNorm(d_model)
#         self.ln_2 = torch.nn.LayerNorm(d_model)
#
#     def forward(self, x, mask):
#         """
#         Compute the encoder layer.
#
#         x dimensions are: (batch_size, sequence_length, d_model)
#         mask dimensions are: (batch_size, sequence_length)
#         mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
#         """
#
#         # Layer normalization 1
#         norm_x = self.ln_1(x)
#
#         # Multi headed self attention
#         attention_output = self.mhsa(norm_x, mask)
#
#         # Residual output
#         residual_output = x + attention_output
#
#         # Layer normalization 2
#         norm_residual_output = self.ln_2(residual_output)
#
#         # Feed forward
#         ff_output = self.ff(norm_residual_output)
#
#         # Dropout
#         if self.training:
#             ff_output = self.dropout(ff_output)
#
#         # Residual output
#         return residual_output + ff_output
#
#
# class DecoderStack(torch.nn.Module):
#     """
#     The decoder stack consists of multiple decoder layers in sequence.
#     """
#
#     def __init__(
#         self,
#         d_model,
#         num_layers,
#         num_heads,
#         d_hid,
#         dropout,
#         max_sequence_length,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.d_hid = d_hid
#         self.dropout = dropout
#         self.max_sequence_length = max_sequence_length
#
#         # Create the encoder layers
#         self.encoder_layers = torch.nn.ModuleList(
#             [DecoderLayer(d_model, num_heads, d_hid, dropout) for _ in range(num_layers)]
#         )
#
#     def forward(self, x, mask):
#         decoder_outputs = x
#         for decoder_layer in self.encoder_layers:
#             decoder_outputs = decoder_layer(decoder_outputs, mask)
#
#         return decoder_outputs
#
#
# class LMHead(torch.nn.Module):
#     """
#     Pytorch module for the language model head.
#     The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
#     """
#
#     def __init__(self, embedding_dimension, number_of_tokens):
#         super().__init__()
#         self.embedding_dimension = embedding_dimension
#         self.number_of_tokens = number_of_tokens
#         self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)
#
#     def forward(self, x):
#         """
#         Compute the language model head.
#
#         x dimensions are: (batch_size, sequence_length, embedding_dimension)
#         output dimensions are: (batch_size, sequence_length, number_of_tokens)
#         """
#         # Compute the linear layer
#         # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
#         linear_output = self.linear(x)
#
#         return linear_output
#
#
# class TransformerLM(SpLightningModule):
#     """
#     Pytorch module for a language model.
#     """
#
#     def __init__(
#         self,
#         sp_tokenizer_file_name: str,
#         d_model: int = 512,  # The dimension of the token embeddings
#         num_layers: int = 6,  # The number of decoder layers to use
#         num_heads: int = 4,  # The number of attention heads to use
#         d_hid: int = None,  # The dimension of the feed forward layer
#         dropout: float = 0.1,  # The dropout rate to use
#         max_sequence_length: int = 512,  # The maximum sequence length to use for attention
#         learning_rate: float = 1e-5,
#     ):
#         super().__init__(sp_tokenizer_file_name)
#         self.vocab_size = self.tokenizer.vocab_size()
#         self.max_sequence_length = max_sequence_length
#         self.d_model = d_model
#         self.number_of_layers = num_layers
#         self.num_heads = num_heads
#
#         if d_hid is None:
#             # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
#             self.d_hid = d_model * 4
#         else:
#             self.d_hid = d_hid
#
#         self.dropout = dropout
#
#         # Create the token embedding layer
#         self.embedding = nn.Embedding(self.vocab_size, self.d_model)
#
#         # Create the positional encoding layer
#         self.pos_enc = PositionalEncoding(d_model, max_sequence_length, self.device)
#
#         # Create the normalization layer
#         self.ln = torch.nn.LayerNorm(d_model)
#
#         # Create the decoder stack
#         self.decoder = DecoderStack(
#             d_model=d_model,
#             num_layers=num_layers,
#             num_heads=num_heads,
#             d_hid=self.d_hid,
#             dropout=dropout,
#             max_sequence_length=max_sequence_length,
#         )
#
#         # Create the language model head
#         self.linear = nn.Linear(d_model, self.vocab_size)
#         self.learning_rate = learning_rate
#         self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
#
#         self.save_hyperparameters()
#
#     def training_step(self, batch, batch_idx):  # pylint: disable=W0613
#         x, y, mask = batch
#         out = self(x, mask)
#         loss = self.loss(out.transpose(1, 2), y)
#
#         self.log("train_loss", loss, prog_bar=True)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):  # pylint: disable=W0613
#         x, y, mask = batch
#
#         out = self(x, mask)
#         loss = self.loss(out.transpose(1, 2), y)
#
#         perplexity = torch.exp(loss).item()
#         self.log_dict({"eval_loss": loss, "eval_pp": perplexity}, prog_bar=True)
#
#     def forward(self, x, mask):
#         # Compute the token embeddings
#         # token_embeddings dimensions are: (batch_size, sequence_length, d_model)
#         token_embeddings = self.embedding(x)
#
#         # Compute the positional encoding
#         # positional_encoding dimensions are: (batch_size, sequence_length, d_model)
#         positional_encoding = self.pos_enc(token_embeddings)
#
#         # Post embedding layer normalization
#         # normalized dimensions are : (batch_size, sequence_length, d_model)
#         positional_encoding_normalized = self.ln(positional_encoding)
#
#         # output of decoder dimensions are (batch_size, sequence_length, d_model)
#         decoder_outputs = self.decoder(positional_encoding_normalized, mask)
#
#         # linear output (batch_size, sequence_length, vocab_size)
#         logits = self.linear(decoder_outputs)
#
#         return logits
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         #     optimizer, T_max=self.trainer.estimated_stepping_batches, verbose=True
#         # )
#         # print(self.trainer.estimated_stepping_batches)
#         # scheduler = NoamAnnealing(
#         #     optimizer,
#         #     d_model=self.d_model,
#         #     warmup_ratio=0.1,
#         #     max_steps=self.trainer.max_steps,
#         #     min_lr=self.learning_rate / 100,
#         # )
#         return [optimizer]  # , [scheduler]
#
#     def generate_next(self, x, mask):
#         """
#         Calculate the token probabilities for the next token in the sequence.
#         """
#         logits = self(x, mask)[:, -1]
#
#         probabilities = torch.softmax(logits, dim=-1)
#
#         return probabilities
#
#     def generate_sequence(self, sample, max_len):
#         self.eval()
#         with torch.no_grad():
#             tokenized_sample = (
#                 torch.LongTensor([self.tokenizer.bos_id()] + self.tokenizer.encode_as_ids(sample))
#                 if isinstance(sample, str)
#                 else sample
#             )
#
#             tokenized_sample = tokenized_sample.to(self.device)
#             padding = [self.tokenizer.pad_id()] * (self.max_sequence_length - len(tokenized_sample))
#             tokenized_sample = torch.cat(
#                 [tokenized_sample, torch.tensor(padding, dtype=torch.long, device=self.device)]
#             ).unsqueeze(0)
#             source_tokens = torch.sum((tokenized_sample != self.tokenizer.pad_id())).item()
#             for i in range(source_tokens, source_tokens + max_len):
#                 mask = torch.ones_like(tokenized_sample)
#                 mask[tokenized_sample == self.tokenizer.pad_id()] = 0
#                 probabilities = self(tokenized_sample, mask).softmax(-1)[:, i - 1]
#                 next_word = torch.multinomial(probabilities, num_samples=1)  # probabilities.argmax()  #
#                 if next_word.item() == self.tokenizer.eos_id():
#                     break
#
#                 tokenized_sample[:, i] = next_word.item()
#         self.train()
#         return self.tokenizer.decode_ids(tokenized_sample.squeeze(0).tolist())


import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim.lr_scheduler import _LRScheduler

from sp_lightning_module import SpLightningModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerLM(SpLightningModule):
    def __init__(
        self,
        sp_tokenizer_file_name,
        d_model: int,
        num_heads: int,
        d_hid: int,
        num_layers: int,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        max_sequence_length: int = 512,
    ):
        super().__init__(sp_tokenizer_file_name)
        self.save_hyperparameters()
        self.max_sequence_length = max_sequence_length
        self.vocab_size = self.tokenizer.vocab_size()
        self.d_model = d_model
        self.num_head = num_heads
        self.d_hid = d_hid
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.pos_encoder = nn.Embedding(self.vocab_size, self.d_model)  # PositionalEncoding(self.d_model, self.dropout)
        decoder_layers = TransformerEncoderLayer(
            self.d_model, self.num_head, self.d_hid, self.dropout, batch_first=True
        )
        self.transformer_decoder = TransformerEncoder(decoder_layers, self.num_layers)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.ln = nn.LayerNorm(self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
        for name, param in self.transformer_decoder.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: Tensor, src_mask: Tensor = None, padding_mask: Tensor = None) -> Tensor:
        # SRC BxL
        x = src
        src_mask = self.generate_square_subsequent_mask(src.shape[-1]).to(src.device)
        src = self.embedding(x) * math.sqrt(self.d_model)  # B x L x d_model
        src = src + self.pos_encoder(x)  # B x L x d_model
        # src = src.permute(1, 0, 2)  # L x B x d_model
        if padding_mask is not None:
            output = self.transformer_decoder(
                src=src, mask=src_mask, src_key_padding_mask=padding_mask
            )  # L x B x d_model
        else:
            output = self.transformer_decoder(src=src, mask=src_mask)  # L x B x d_model
        output = self.ln(output)  # B x L x vocab_size
        output = self.linear(output)  # B x L x vocab_size
        return output

    def training_step(self, batch, batch_idx):  # pylint: disable=W0613
        src, tgt, pad_mask = batch
        out = self(src, padding_mask=pad_mask)
        loss = self.loss(out.view(-1, self.vocab_size), tgt.view(-1))

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=W0613
        src, tgt, pad_mask = batch

        out = self(src, padding_mask=pad_mask)
        loss = self.loss(out.view(-1, self.vocab_size), tgt.view(-1))

        perplexity = torch.exp(loss).item()
        self.log_dict({"eval_loss": loss, "eval_pp": perplexity}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, fused=True)
        scheduler = NoamAnnealing(
            optimizer,
            d_model=self.d_model,
            warmup_ratio=0.25,
            max_steps=self.trainer.max_steps,
            min_lr=self.learning_rate / 100,
        )
        return optimizer  # [scheduler]

    def generate_sequence(self, sample, max_len):
        with torch.no_grad():
            tokenized_sample = (
                torch.LongTensor([self.tokenizer.bos_id()] + self.tokenizer.encode_as_ids(sample))
                if isinstance(sample, str)
                else sample
            )

            result_ids = tokenized_sample.tolist()
            tokenized_sample = tokenized_sample.to(self.device)

            for _ in range(max_len):
                next_word = self(tokenized_sample)[-1].argmax(-1)
                result_ids.append(next_word.item())

                if next_word.item() == self.tokenizer.eos_id():
                    break

                tokenized_sample = torch.cat([tokenized_sample, next_word.unsqueeze(0)], dim=0)

        return self.tokenizer.decode_ids(result_ids)


class NoamAnnealing(_LRScheduler):
    def __init__(
        self, optimizer, *, d_model, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        print(self.base_lrs)

    def get_lr(self):
        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self._noam_annealing(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        else:
            mult = self._normalize * step ** (-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr
