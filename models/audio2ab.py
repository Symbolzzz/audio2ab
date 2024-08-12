import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple
import os
import pickle
import numpy as np
from torch.nn.utils import weight_norm
from .utils.build_vocab import Vocab
import math

class JitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)

# ----------------------------------------------------------------------------------------------------------------------
class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden

# ----------------------------------------------------------------------------------------------------------------------
class JitGRU(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True):
        super(JitGRU, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)] + [JitGRULayer(JitGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            #print(pre_trained_embedding.shape)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, 128, 15, 6,  first_dilation=0,downsample=True),     
            )
        
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1) 
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2) 

# TODO：加一个语义编码器

class PoseGenerator(nn.Module):
    """
    End2End model
    audio, text and speaker ID encoder are customized based on Yoon et al. SIGGRAPH ASIA 2020
    """
    def __init__(self, args):
        super().__init__()
        self.pre_length = args.pre_frames 
        self.gen_length = args.pose_length - args.pre_frames
        self.pose_dims = args.pose_dims
        # 不需要face
        self.facial_f = args.facial_f
        self.speaker_f = args.speaker_f
        # 加入sem的输出维度 8
        self.semantic_f = args.semantic_f
        self.audio_f = args.audio_f
        self.word_f = args.word_f
        self.emotion_f = args.emotion_f
        self.facial_dims = args.facial_dims
        self.speaker_dims = args.speaker_dims
        # 加入sem输入维度 30
        self.semantic_dims = args.semantic_dims
        self.emotion_dims = args.emotion_dims
        
        # 输入维度需要改 增加 facial_dims
        self.in_size = self.audio_f + self.pose_dims + self.facial_dims + self.word_f + 2
        self.audio_encoder = WavEncoder(self.audio_f)
        self.hidden_size = args.hidden_size
        self.n_layer = args.n_layer

        # 不需要facial_encoder
        # if self.facial_f is not 0:  
        #     self.facial_encoder = nn.Sequential( 
        #         BasicBlock(self.facial_dims, self.facial_f//2, 7, 1, first_dilation=3,  downsample=True),
        #         BasicBlock(self.facial_f//2, self.facial_f//2, 3, 1, first_dilation=1,  downsample=True),
        #         BasicBlock(self.facial_f//2, self.facial_f//2, 3, 1, first_dilation=1, ),
        #         BasicBlock(self.facial_f//2, self.facial_f, 3, 1, first_dilation=1,  downsample=True),   
        #     )
        # else:
        #     self.facial_encoder = None

        self.text_encoder = None   
        if self.word_f is not 0:
            with open(f"{args.root_path}{args.train_data_path[:-6]}vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                pre_trained_embedding = self.lang_model.word_embedding_weights
            self.text_encoder = TextEncoderTCN(args, args.word_index_num, args.word_dims, pre_trained_embedding=pre_trained_embedding,
                                           dropout=args.dropout_prob)

        # 不需要speaker embedding
        # The initial representation of speaker ID and emotion are both one-hot vectors, as vID ∈ R30 and vE ∈ R8. Follow the suggestion in [52], we use embedding-layer as speaker ID encoder, EID. As the speaker ID does not change instantly, we only use the current frame speaker ID to calculate its latent features.
        self.speaker_embedding = None
        if self.speaker_f is not 0:
            self.in_size += self.speaker_f
            self.speaker_embedding =   nn.Sequential(
                nn.Embedding(self.speaker_dims, self.speaker_f),
                nn.Linear(self.speaker_f, self.speaker_f), 
                nn.LeakyReLU(0.1, True)
            )
        
        # 加入semantic embedding
        # The initial representations of the semantic relevance label vS ∈ R30 and emotion label vE ∈ 8 are both one-hot vectors. We used an embedding layer as the semantic relevance label encoder E S , because only the semantic relevance label of the current frame was used to calculate its latent feature.
        # self.semantic_embedding = None
        # if self.semantic_f is not 0:
        #     self.in_size += self.semantic_f
        #     self.semantic_embedding = nn.Sequential(
        #         nn.Embedding(self.semantic_dims, self.semantic_f),
        #         nn.Linear(self.semantic_f, self.semantic_f),
        #         nn.LeakyReLU(0.1, True)
        #     )

            
        self.emotion_embedding = None
        if self.emotion_f is not 0:
            self.in_size += self.emotion_f
            
            self.emotion_embedding =   nn.Sequential(
                nn.Embedding(self.emotion_dims, self.emotion_f),
                nn.Linear(self.emotion_f, self.emotion_f) 
            )

            self.emotion_embedding_tail = nn.Sequential( 
                nn.Conv1d(self.emotion_f, 8, 9, 1, 4),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(8, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, 16, 9, 1, 4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv1d(16, self.emotion_f, 9, 1, 4),
                nn.BatchNorm1d(self.emotion_f),
                nn.LeakyReLU(0.3, inplace=True),
            )
        
        # 不需要 LSTM， 使用 GRU
        self.GRU = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.hidden_size//2, 27)
        )
        
        # 不需要 LSTM， 使用 GRU，这里为什么是 +27 而不是 + 256？
        self.GRU_hands = nn.GRU(self.in_size+27, hidden_size=self.hidden_size, num_layers=args.n_layer, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out_hands = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.hidden_size//2, 141-27) # 这个维度好像有问题
        )
        
        # 增加 facial decoder
        self.GRU_face = nn.GRU(self.in_size + 27 + 114, hidden_size=self.facial_dims + 1, num_layers=args.n_layer, batch_first=True,
                            bidirectional=True, dropout=args.dropout_prob)
        self.out_face = nn.Sequential(
            nn.Linear(self.facial_dims + 1, (self.facial_dims + 1) // 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear((self.facial_dims + 1) // 2, 51) # 按照论文的描述，输出维度是 v^F ∈ R^17 * 3，这里应该怎么改？
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
            

    def forward(self, pre_seq, in_audio=None, in_facial=None,  in_text=None, in_id = None, in_emo=None, is_test=False):
        if self.do_flatten_parameters:
            # 不需要 LSTM，改成 GRU
            self.GRU.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if in_audio is not None:
            audio_feat_seq = self.audio_encoder(in_audio) 
        if in_text is not None:
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])
        
        # 不需要 facial
        # if self.facial_f is not 0:
        #     face_feat_seq = self.facial_encoder(in_facial.permute([0, 2, 1]))
        #     face_feat_seq = face_feat_seq.permute([0, 2, 1])
        
        # 不需要 speaker embedding
        speaker_feat_seq = None
        if self.speaker_embedding: 
            speaker_feat_seq = self.speaker_embedding(in_id)
        
        # 加入 semantic embedding
        # semantic_feat_seq = None
        # if self.semantic_embedding:
        #     semantic_feat_seq = self.semantic_embedding(in_sem)
            
        emo_feat_seq = None
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(in_emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])

        if  audio_feat_seq.shape[1] != pre_seq.shape[1]:
            diff_length = pre_seq.shape[1] - audio_feat_seq.shape[1]
            audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-diff_length:, :].reshape(1,diff_length,-1)),1)
       
        # 不需要 facial
        # if self.audio_f is not 0 and self.facial_f is 0:
        #     in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        # elif self.audio_f is not 0 and self.facial_f is not 0:
        #     in_data = torch.cat((pre_seq, audio_feat_seq, face_feat_seq), dim=2)
        # else: pass
        
        # 直接将 audio feat 加入 in_data 就好了
        if self.audio_f is not 0:
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        
        if text_feat_seq is not None:
            in_data = torch.cat((in_data, text_feat_seq), dim=2)
        if emo_feat_seq is not None:
            in_data = torch.cat((in_data, emo_feat_seq), dim=2)
        
        # 不需要 speaker feat
        if speaker_feat_seq is not None:
            repeated_s = speaker_feat_seq
            if len(repeated_s.shape) == 2:
                repeated_s = repeated_s.reshape(1, repeated_s.shape[1], repeated_s.shape[0])
            repeated_s = repeated_s.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_s), dim=2)
            
        # 加入 semantic feat
        # if semantic_feat_seq is not None:
        #     repeated_s = semantic_feat_seq
        #     if len(repeated_s.shape) == 2:
        #         repeated_s = repeated_s.reshape(1, repeated_s.shape[1], repeated_s.shape[0])
        #     repeated_s = repeated_s.repeat(1, in_data.shape[1], 1)
        #     in_data = torch.cat((in_data, repeated_s), dim=2)
        
        # 不需要 LSTM，改成 GRU
        # 需要将原始的 facial feat 放入编码，x^M = x^T * x^S * x^E * x^A * v^B * v^G * v^F
        if in_facial is not None:
            in_data = torch.cat((pre_seq, in_facial), dim=2)
        output, _ = self.GRU(in_data)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        return decoder_outputs
    

class Audio2AB(PoseGenerator):
    def __init__(self, args):
        super().__init__(args) 
        self.audio_fusion_dim = self.audio_f+self.speaker_f+self.emotion_f+self.word_f
        self.facial_fusion_dim = self.audio_fusion_dim + self.facial_f
        self.audio_fusion = nn.Sequential(
            nn.Linear(self.audio_fusion_dim, self.hidden_size//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.hidden_size//2, self.audio_f),
            nn.LeakyReLU(0.1, True),
        )
        
        self.facial_fusion = nn.Sequential(
            nn.Linear(self.facial_fusion_dim, self.hidden_size//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.hidden_size//2, self.facial_f),
            nn.LeakyReLU(0.1, True),
        )
        
    def forward(self, pre_seq, pre_seq_facial, in_audio=None, in_text=None, in_id = None, in_emo=None):
        if self.do_flatten_parameters:
            # 不需要 LSTM， 使用 GRU
            self.GRU.flatten_parameters()
            
        decoder_hidden = decoder_hidden_hands = None
        # 删掉 speaker_feat_seq 和 face_feat_seq
        text_feat_seq = audio_feat_seq = emo_feat_seq  =  speaker_feat_seq = None
        in_data = None
        
        # 不需要speaker embedding
        if self.speaker_embedding: 
            speaker_feat_seq = self.speaker_embedding(in_id)
            if len(speaker_feat_seq.shape) == 2:
                speaker_feat_seq = speaker_feat_seq.reshape(1, speaker_feat_seq.shape[1], speaker_feat_seq.shape[0])
            speaker_feat_seq = speaker_feat_seq.repeat(1, pre_seq.shape[1], 1)
            in_data = torch.cat((in_data, speaker_feat_seq), 2) if in_data is not None else speaker_feat_seq
            
        # 加入semantic embedding，可能需要改改
        # if self.semantic_embedding:
        #     # 潜入层需要长整型
        #     # in_sem = in_sem.long()
        #     # sem 的范围在0-1之间
        #     in_sem = (in_sem * 10).long()
        #     semantic_feat_seq = self.semantic_embedding(in_sem)
        #     if len(semantic_feat_seq.shape) == 2:
        #         semantic_feat_seq = semantic_feat_seq.reshape(1, semantic_feat_seq.shape[1], semantic_feat_seq.shape[0])
        #     # 不需要 repeat
        #     # semantic_feat_seq = semantic_feat_seq.repeat(1, pre_seq.shape[1], 1)
        #     semantic_feat_seq = semantic_feat_seq.repeat(1, 1, 1)
        #     in_data = torch.cat((in_data, semantic_feat_seq), 2) if in_data is not None else semantic_feat_seq
            
        # 调试用
        # print("in_data shape:", in_data.shape)
            
        # 输出维度由 8 变成 128
        if self.emotion_embedding:
            emo_feat_seq = self.emotion_embedding(in_emo)
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            emo_feat_seq = self.emotion_embedding_tail(emo_feat_seq) 
            emo_feat_seq = emo_feat_seq.permute([0,2,1])
            # 调试用
            # print("emo_feat_seq shape:", emo_feat_seq.shape)
            in_data = torch.cat((in_data, emo_feat_seq), 2) if in_data is not None else emo_feat_seq
            
        if in_text is not None:
            text_feat_seq, _ = self.text_encoder(in_text)
            in_data = torch.cat((in_data, text_feat_seq), 2) if in_data is not None else text_feat_seq
            
        if in_audio is not None:
            audio_feat_seq = self.audio_encoder(in_audio) 
            if in_text is not None:
                if (audio_feat_seq.shape[1] != text_feat_seq.shape[1]):
                    min_gap = text_feat_seq.shape[1] - audio_feat_seq.shape[1]
                    audio_feat_seq = torch.cat((audio_feat_seq, audio_feat_seq[:,-min_gap:, :]),1)
                
            # 需要把 speaker 改成 semantic
            audio_fusion_seq = self.audio_fusion(torch.cat((audio_feat_seq, emo_feat_seq, speaker_feat_seq, text_feat_seq), dim=2).reshape(-1, self.audio_fusion_dim))
            audio_feat_seq = audio_fusion_seq.reshape(*audio_feat_seq.shape)
            in_data = torch.cat((in_data, audio_feat_seq), 2) if in_data is not None else audio_feat_seq
        
        
        # 不需要 face encoder
        # if self.facial_f is not 0:
        #     face_feat_seq = self.facial_encoder(in_facial.permute([0, 2, 1]))
        #     face_feat_seq = face_feat_seq.permute([0, 2, 1])
        #     if (audio_feat_seq.shape[1] != face_feat_seq.shape[1]):
        #         min_gap_2 = face_feat_seq.shape[1] - audio_feat_seq.shape[1]
        #         if min_gap_2 > 0:
        #             face_feat_seq = face_feat_seq[:,:audio_feat_seq.shape[1], :]
        #         else:
        #             face_feat_seq = torch.cat((face_feat_seq, face_feat_seq[:,-min_gap_2:, :]),1)
                
        #     face_fusion_seq = self.facial_fusion(torch.cat((face_feat_seq, audio_feat_seq, emo_feat_seq, speaker_feat_seq, text_feat_seq), dim=2).reshape(-1, self.facial_fusion_dim))
        #     face_feat_seq = face_fusion_seq.reshape(*face_feat_seq.shape)
        #     in_data = torch.cat((in_data, face_feat_seq), 2) if in_data is not None else face_feat_seq
            
        
        # 在解码器中加入 facial feat
        # X^B
        in_data = torch.cat((pre_seq_facial, in_data), dim=2)
        in_data = torch.cat((pre_seq, in_data), dim=2)
        output, _ = self.GRU(in_data)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] 
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        
        # X^G
        in_data = torch.cat((in_data, decoder_outputs), dim=2)
        output_hands, _ = self.GRU_hands(in_data)
        output_hands = output_hands[:, :, :self.hidden_size] + output_hands[:, :, self.hidden_size:]
        output_hands = self.out_hands(output_hands.reshape(-1, output_hands.shape[2]))
        decoder_outputs_hands = output_hands.reshape(in_data.shape[0], in_data.shape[1], -1)
        
        # TODO: 增加 X^F 
        in_data = torch.cat((in_data, decoder_outputs_hands), dim=2)
        output_face, _ = self.GRU_face(in_data)
        output_face = output_face[:, :, :self.facial_dims + 1] + output_face[:, :, self.facial_dims + 1:]
        output_face = self.out_face(output_face.reshape(-1, output_face.shape[2]))
        decoder_outputs_face = output_face.reshape(in_data.shape[0], in_data.shape[1], -1)
        
        
        decoder_outputs_final = torch.zeros((in_data.shape[0], in_data.shape[1], 141 + 51)).cuda()
        decoder_outputs_final[:, :, 0:18] = decoder_outputs[:, :, 0:18]
        decoder_outputs_final[:, :, 18:75] = decoder_outputs_hands[:, :, 0:57]
        decoder_outputs_final[:, :, 75:84] = decoder_outputs[:, :, 18:27]
        decoder_outputs_final[:, :, 84:141] = decoder_outputs_hands[:, :, 57:114]
        # 在文章找不到描述，直接将 facial 加在末尾，可能需要改改
        decoder_outputs_final[:, :, 141:141 + 51] = decoder_outputs_face[:, :, 0:51]
        return decoder_outputs_final

    
# 这是判别器
class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.pose_dims + args.facial_dims

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(8, 8, 3),
        )

        self.GRU = JitGRU(8, hidden_size=self.hidden_size, num_layers=4, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34-6, 1)
       
        # self.do_flatten_parameters = False
        # if torch.cuda.device_count() > 1:
        #     self.do_flatten_parameters = True

    def forward(self, poses):
        # if self.do_flatten_parameters:
        #     self.GRU.flatten_parameters()
        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)
        # 这里可以使用 feat 一个参数
        # Debugging shapes
        # print(f"feat shape: {feat.shape}")
        
        output, _ = self.GRU(feat)
        # print(f"output shape after GRU: {output.shape}")
        
        # Checking if the hidden_size and concatenation work as intended
        # if output.shape[2] != 2 * self.hidden_size:
        #     raise ValueError(f"Expected output shape's third dimension to be {2 * self.hidden_size}, but got {output.shape[2]}")
        # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output
