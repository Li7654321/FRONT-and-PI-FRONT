#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model neural network structure
env: python 3.12 + pytorch 2.7.0 + CUDA 12.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from physics_layers import WaveformSynthesizer


# ==============================================================================
# 1. Basic Graph Convolution Layer (Dual-Path Graph Fusion + Graph Residual Connection)
# ==============================================================================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.weight_spatial = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_adaptive = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_residual = nn.Parameter(torch.Tensor(in_channels, out_channels))

        nn.init.xavier_uniform_(self.weight_spatial)
        nn.init.xavier_uniform_(self.weight_adaptive)
        nn.init.xavier_uniform_(self.weight_residual)

    def forward(self, x, adj, adp_matrix):
        x_permuted = x.permute(0, 2, 3, 1)  # (B, T, N, C)

        x_spatial = torch.einsum('bij,bjtc->bitc', adj, x_permuted)
        out_spatial = torch.matmul(x_spatial, self.weight_spatial)

        x_adaptive = torch.einsum('ij,bjtc->bitc', adp_matrix, x_permuted)
        out_adaptive = torch.matmul(x_adaptive, self.weight_adaptive)

        out_residual = torch.matmul(x_permuted, self.weight_residual)
        out = out_spatial + out_adaptive + out_residual

        return out.permute(0, 3, 1, 2)


# ==============================================================================
# 2. Fault space layer (Fault GAT vs Fault GCN)
# ==============================================================================
# 2.1 Slip: GAT layer
class FaultGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.2):
        super(FaultGATLayer, self).__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, fault_adj=None):
        Wh = self.W(x)
        a_input_i = torch.matmul(Wh, self.a[:self.W.out_features, :])
        a_input_j = torch.matmul(Wh, self.a[self.W.out_features:, :])
        e = self.leakyrelu(a_input_i + a_input_j.transpose(1, 2))

        if fault_adj is None:
            N = e.size(1)
            idx = torch.arange(N, device=e.device)
            dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            fault_adj = (dist <= 5).float()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(fault_adj > 0, e, zero_vec)

        tau = 0.2
        attention = F.softmax(attention / tau, dim=2)      
        attention = F.dropout(attention, p=0.3, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


# 2.2  Rupture Time: GCN layer
class FaultGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FaultGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, fault_adj):
        hw = torch.matmul(x, self.weight)

        if fault_adj.dim() == 2:
            adj = fault_adj.unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            adj = fault_adj

        rowsum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_adj = adj / rowsum

        out = torch.bmm(norm_adj, hw)
        return F.elu(out)


# ==============================================================================
# 3. STGCN Block
# ==============================================================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.t_conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.s_gcn = GraphConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.t_conv2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, adj, adp_matrix):
        res = self.residual(x)
        x = F.leaky_relu(self.bn1(self.t_conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.s_gcn(x, adj, adp_matrix)), 0.1)
        x = self.bn3(self.t_conv2(x))
        return F.leaky_relu(x + res, 0.1)


# ==============================================================================
# 4. Model Structure
# ==============================================================================
class PhysicsInformedSTGNN(nn.Module):
    def __init__(self, params, gf_tensor=None):
        super(PhysicsInformedSTGNN, self).__init__()
        dt_wave = params.get('dt_wave', 0.5)
        max_vals = params.get('max_vals', {'slip': 20.0, 'rise': 35.0, 'rupt': 260.0})
        self.num_faults = params.get('Nfault', 985)
        gf_shape = gf_tensor.shape if gf_tensor is not None else \
                                               (self.num_faults, 2, num_stations * 3, params.get('Tlen', 512))
        num_stations = params.get('Nstan', 60)
        embed_dim = 10
        self.node_emb1 = nn.Parameter(torch.randn(num_stations, embed_dim), requires_grad=True)
        self.node_emb2 = nn.Parameter(torch.randn(embed_dim, num_stations), requires_grad=True)
        nn.init.xavier_uniform_(self.node_emb1)
        nn.init.xavier_uniform_(self.node_emb2)

        # Encoder
        self.input_conv = nn.Conv2d(3, 32, (1, 5), padding=(0, 2))
        self.stgcn1 = STGCNBlock(32, 64)
        self.pool1 = nn.AvgPool2d((1, 2))
        self.stgcn2 = STGCNBlock(64, 128)
        self.pool2 = nn.AvgPool2d((1, 2))
        self.stgcn3 = STGCNBlock(128, 256)
        self.pool3 = nn.AvgPool2d((1, 2))

        # ======================================================================
        # Physical Projection Layer
        # ======================================================================
        self.time_compress = nn.AdaptiveMaxPool2d((num_stations, 4))
        self.mag_branch = nn.Sequential(
            nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.channel_reduce = nn.Conv1d(1024, 64, 1)
        
        self.station_to_fault = nn.Linear(num_stations, self.num_faults)

        # ======================================================================
        # Source Branch
        # ======================================================================
        self.src_branch_conv = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.src_branch_gat = FaultGATLayer(in_channels=64, out_channels=64)

        self.out_ss = nn.Conv1d(64, 1, 1)
        self.out_ds = nn.Conv1d(64, 1, 1)
        self.out_rise = nn.Conv1d(64, 1, 1)

        # ======================================================================
        # Rupture Branch
        # ======================================================================
        self.rup_conv1 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rup_gcn = FaultGCNLayer(in_channels=128, out_channels=64)
        self.out_rup_raw = nn.Conv1d(64, 1, 1)

        # Physical Forward Synthesizer
        self.waveform_synth = WaveformSynthesizer(gf_shape=gf_shape, dt=dt_wave, max_vals=max_vals, gf_tensor=gf_tensor)

    def forward(self, wave, adj, fault_adj=None):
        # 1. Waveform Feature Extraction
        x = self.input_conv(wave)
        adp_matrix = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2)), dim=1)
        x = self.pool1(self.stgcn1(x, adj, adp_matrix))
        x = self.pool2(self.stgcn2(x, adj, adp_matrix))
        x = self.pool3(self.stgcn3(x, adj, adp_matrix)) 

        # ======================================================================
        # 2. Spatial Projection and Decoupling
        # ======================================================================
        x_comp = self.time_compress(x)
        x_spatial = x_comp.transpose(2, 3).reshape(x_comp.size(0), -1, x_comp.size(2))

        x_global = torch.mean(x_spatial, dim=2)
        out_mag = torch.sigmoid(self.mag_branch(x_global))

        x_reduced = self.channel_reduce(x_spatial)
        bottleneck = self.station_to_fault(x_reduced)  

        # ======================================================================
        # 3. Source Branch：GAT
        # ======================================================================
        feat_conv = self.src_branch_conv(bottleneck)  
        bottleneck_t = bottleneck.transpose(1, 2)  
        feat_gat = self.src_branch_gat(bottleneck_t, fault_adj=fault_adj).transpose(1, 2)

        src_feat = feat_conv + feat_gat
        ss_pred = torch.tanh(self.out_ss(src_feat))
        ds_pred = torch.tanh(self.out_ds(src_feat))
        rise_pred = torch.sigmoid(self.out_rise(src_feat))

        out_source = torch.cat([ss_pred, ds_pred, rise_pred], dim=1)

        # ======================================================================
        # 4. Rupture Branch：GCN
        # ======================================================================
        rup_feat = self.rup_conv1(bottleneck)  
        rup_feat_t = rup_feat.transpose(1, 2)  
        rup_gcn_feat = self.rup_gcn(rup_feat_t, fault_adj=fault_adj).transpose(1, 2) 

        out_rup = torch.sigmoid(self.out_rup_raw(rup_gcn_feat))

        # ======================================================================
        # 5. Forward modeling output
        # ======================================================================
        waveform_out = self.waveform_synth(out_source, out_rup)

        return {
            'source_output': out_source,
            'rupt_output': out_rup,
            'waveform_output': waveform_out,
            'mag_output': out_mag
        }