"""
model.py
========
PolicyNet: predicts the next dial (AOI) + saccade/duration from:
  - past fixation AOI sequence
  - recent signal history as frame×dial tokens

Architecture:
  - Fixation branch:
      * AOI embedding
      * GRU over past fixations
  - Signal branch:
      * one token per (frame, dial)
      * token = raw signal features + dial embedding + time embedding
      * multi-head cross-attention pooling:
            query  = fixation history representation
            keys   = signal tokens
            values = signal tokens
      * outputs a single signal summary vector
  - Fusion:
      * concatenate fixation summary + signal summary
      * predict:
            - next dial logits
            - [saccade_norm, duration_norm]

Inputs:
    past_aois : LongTensor  [B, N]       — AOI labels (0=pad, 1-6=dials)
    signal    : FloatTensor [B, F, 6, C] — signal features per (frame, dial): sin, cos, urgency, distance

Outputs:
    logits    : FloatTensor [B, 6]
    temporal  : FloatTensor [B, 2]
"""

import torch
import torch.nn as nn
import config


class PolicyNet(nn.Module):
    def __init__(
        self,
        n_aoi:              int   = 6,
        emb_dim:            int   = config.EMB_DIM,
        sig_feat:           int   = config.SIG_FEAT,
        hidden:             int   = config.HIDDEN_DIM,
        dropout:            float = config.DROPOUT,
        fix_branch_dropout: float = config.FIX_BRANCH_DROPOUT,
        use_fixations:      bool  = True,
        use_signal:         bool  = True,
    ):
        super().__init__()
        self.n_aoi = n_aoi
        self.hidden = hidden
        self.sig_feat = sig_feat
        self.use_fixations = use_fixations
        self.use_signal = use_signal
        self.fix_branch_dropout = fix_branch_dropout
        self.max_signal_frames = config.SIGNAL_FRAMES

        # Number of attention heads for signal readout
        self.n_heads = getattr(config, "NHEAD", 4)
        if hidden % self.n_heads != 0:
            raise ValueError(
                f"HIDDEN_DIM ({hidden}) must be divisible by NHEAD ({self.n_heads})."
            )
        self.head_dim = hidden // self.n_heads

        # ── Fixation encoder ────────────────────────────────────────────────
        self.aoi_emb       = nn.Embedding(n_aoi + 1, emb_dim, padding_idx=0)
        self.temporal_proj = nn.Linear(2, emb_dim)        # duration_norm + saccade_norm → emb
        self.fix_input_proj = nn.Linear(emb_dim * 2, emb_dim)  # cat([aoi, temporal]) → emb
        self.fix_gru       = nn.GRU(emb_dim, hidden, batch_first=True)

        # ── Signal token construction ───────────────────────────────────────
        self.dial_emb = nn.Embedding(n_aoi + 1, emb_dim, padding_idx=0)
        self.time_emb = nn.Embedding(self.max_signal_frames, emb_dim)

        token_in_dim = sig_feat + emb_dim + emb_dim

        self.token_mlp = nn.Sequential(
            nn.Linear(token_in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ── Transformer encoder over signal tokens ──────────────────────────
        _enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.n_heads,
            dim_feedforward=getattr(config, 'FF_DIM', hidden),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.signal_transformer = nn.TransformerEncoder(
            _enc_layer,
            num_layers=getattr(config, 'N_SIGNAL_LAYERS', 1),
        )

        # ── Multi-head cross-attention pooling ──────────────────────────────
        # Conditioned path: query = GRU hidden state (behavioural context)
        self.q_proj    = nn.Linear(hidden, hidden)
        self.out_proj  = nn.Linear(hidden, hidden)
        # Global path: query = learned parameter (pure urgency-driven)
        self.q_proj_global   = nn.Linear(hidden, hidden)
        self.out_proj_global = nn.Linear(hidden, hidden)
        # Shared key/value projections (same signal tokens for both paths)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)

        self.attn_dropout = nn.Dropout(dropout)

        # Learned global query (urgency-driven, no fixation bias)
        self.global_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.global_query, std=0.02)

        # sig_fc fuses both signal summaries [conditioned + global] → hidden
        self.sig_fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Fusion ──────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Output heads ────────────────────────────────────────────────────
        self.dial_head = nn.Linear(hidden, n_aoi)

        # Future signal encoder: mean-pool F_future frames → [B, 6*sig_feat] → hidden
        self.future_sig_fc = nn.Sequential(
            nn.Linear(6 * sig_feat, hidden),
            nn.ReLU(),
        )

        # Temporal head conditioned on:
        #   fused (hidden) + chosen dial embedding (emb_dim) + future signal (hidden)
        # outputs [sacc_mu, dur_mu, sacc_log_sigma, dur_log_sigma]
        self.temporal_head = nn.Linear(hidden + emb_dim + hidden, 4)

    def encode_fixations(self, past_aois, past_temporal):
        """
        past_aois:    [B, N]
        past_temporal:[B, N, 2]  — [duration_norm, saccade_norm] per past fixation
        returns:      [B, H]
        """
        B = past_aois.size(0)
        N = past_aois.size(1)
        device = past_aois.device

        if not self.use_fixations or N == 0:
            return torch.zeros(B, self.hidden, device=device)

        if self.training and self.fix_branch_dropout > 0:
            mask = torch.rand(B, device=device) < self.fix_branch_dropout
            pa = past_aois.clone()
            pt = past_temporal.clone()
            pa[mask] = 0
            pt[mask] = 0.0
        else:
            pa = past_aois
            pt = past_temporal

        aoi_e = self.aoi_emb(pa)                                    # [B, N, emb_dim]
        tmp_e = self.temporal_proj(pt)                              # [B, N, emb_dim]
        emb   = self.fix_input_proj(torch.cat([aoi_e, tmp_e], dim=-1))  # [B, N, emb_dim]
        _, h  = self.fix_gru(emb)                                   # [1, B, H]
        h_fix = h.squeeze(0)                              # [B, H]
        return h_fix

    def build_signal_tokens(self, signal):
        """
        signal: [B, F, 6, C]
        returns:
            tok_h: [B, T, H], where T = F * 6
        """
        B, F, D, C = signal.shape
        device = signal.device

        if D != self.n_aoi:
            raise ValueError(f"Expected {self.n_aoi} dials, got {D}")
        if C != self.sig_feat:
            raise ValueError(f"Expected SIG_FEAT={self.sig_feat}, got {C}")
        if F > self.max_signal_frames:
            raise ValueError(
                f"Signal has {F} frames, but model max is {self.max_signal_frames}. "
                f"Increase config.SIGNAL_FRAMES or rebuild the dataset for the new scenario."
            )

        # Dial IDs: 1..6
        dial_ids = torch.arange(1, D + 1, device=device).view(1, 1, D).expand(B, F, D)
        # Time IDs: 0..F-1
        time_ids = torch.arange(F, device=device).view(1, F, 1).expand(B, F, D)

        dial_e = self.dial_emb(dial_ids)                          # [B, F, D, E]
        time_e = self.time_emb(time_ids)                          # [B, F, D, E]

        tok = torch.cat([signal, dial_e, time_e], dim=-1)        # [B, F, D, C+2E]
        tok = tok.reshape(B, F * D, -1)                          # [B, T, token_in]
        tok_h = self.token_mlp(tok)                              # [B, T, H]
        return tok_h

    def multi_head_cross_attention(self, q, k, v, out_proj):
        """
        q:        [B, 1, H]
        k:        [B, T, H]
        v:        [B, T, H]
        out_proj: nn.Linear(H, H) — separate per attention path

        returns:
            out: [B, H]
            attn_mean: [B, T]
        """
        B, T, _ = k.shape
        H = self.hidden
        Dh = self.head_dim
        Nh = self.n_heads

        # Project to heads
        q = q.reshape(B, 1, Nh, Dh).transpose(1, 2)   # [B, Nh, 1, Dh]
        k = k.reshape(B, T, Nh, Dh).transpose(1, 2)   # [B, Nh, T, Dh]
        v = v.reshape(B, T, Nh, Dh).transpose(1, 2)   # [B, Nh, T, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)   # [B, Nh, 1, T]
        attn = torch.softmax(scores, dim=-1)                           # [B, Nh, 1, T]
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                                    # [B, Nh, 1, Dh]
        out = out.transpose(1, 2).contiguous().reshape(B, 1, H)        # [B, 1, H]
        out = out_proj(out).squeeze(1)                                 # [B, H]

        # Mean attention across heads for debugging/visualization
        attn_mean = attn.mean(dim=1).squeeze(1)                        # [B, T]
        return out, attn_mean

    def encode_signal(self, signal, h_fix, return_attention=False):
        """
        signal: [B, F, 6, C]
        h_fix:  [B, H]

        returns:
            h_sig: [B, H]
            optionally attn_map: [B, F, 6]
        """
        B, F, D, _ = signal.shape
        device = signal.device

        if not self.use_signal:
            h_sig = torch.zeros(B, self.hidden, device=device)
            if return_attention:
                return h_sig, torch.zeros(B, F, D, device=device)
            return h_sig

        tok_h = self.build_signal_tokens(signal)                       # [B, T, H]
        tok_h = self.signal_transformer(tok_h)                         # [B, T, H]

        # K and V are shared between both attention paths
        k = self.k_proj(tok_h)                                         # [B, T, H]
        v = self.v_proj(tok_h)                                         # [B, T, H]

        # Conditioned path: query from GRU hidden state (behavioural bias)
        q_cond = self.q_proj(h_fix).unsqueeze(1)                       # [B, 1, H]
        pooled_cond, attn_flat = self.multi_head_cross_attention(
            q_cond, k, v, self.out_proj)                               # [B, H], [B, T]

        # Global path: learned query (pure urgency-driven, no fixation bias)
        q_glob = self.q_proj_global(
            self.global_query.expand(B, -1, -1))                       # [B, 1, H]
        pooled_glob, _ = self.multi_head_cross_attention(
            q_glob, k, v, self.out_proj_global)                        # [B, H]

        h_sig = self.sig_fc(torch.cat([pooled_cond, pooled_glob], dim=-1))  # [B, H]

        if return_attention:
            attn_map = attn_flat.reshape(B, F, D)                      # [B, F, 6]
            return h_sig, attn_map

        return h_sig

    def forward(self, past_aois, signal, past_temporal,
                chosen_dial=None, future_signal=None, return_attention=False):
        """
        past_aois:    [B, N]
        signal:       [B, F, 6, C]
        past_temporal:[B, N, 2]   — [duration_norm, saccade_norm] per past fixation
        chosen_dial:  LongTensor [B] — 0-indexed chosen dial (for temporal conditioning)
                      If None, uses pad embedding (zeros) — temporal head receives no dial signal.
        future_signal:[B, F_future, 6, C] — signal during predicted fixation
                      If None, zeros are used — temporal head receives no future context.

        returns by default:
            logits   [B, 6]
            temporal [B, 4]  — [sacc_mu, dur_mu, sacc_log_sigma, dur_log_sigma]

        if return_attention=True:
            logits, temporal, attn_map  (attn_map: [B, F, 6])
        """
        B      = past_aois.size(0)
        device = past_aois.device

        h_fix = self.encode_fixations(past_aois, past_temporal)        # [B, H]

        if return_attention:
            h_sig, attn_map = self.encode_signal(signal, h_fix, return_attention=True)
        else:
            h_sig = self.encode_signal(signal, h_fix, return_attention=False)

        fused  = self.fusion(torch.cat([h_fix, h_sig], dim=-1))        # [B, H]
        logits = self.dial_head(fused)                                  # [B, 6]

        # ── Temporal head: conditioned on chosen dial + future signal ────────
        # Detach fused so NLL gradients don't flow back through the shared
        # representation and interfere with dial head convergence.
        fused = fused.detach()
        # chosen dial embedding (1-indexed; 0 = pad/unknown)
        if chosen_dial is None:
            dial_ids = torch.zeros(B, dtype=torch.long, device=device)
        else:
            dial_ids = (chosen_dial + 1).clamp(0, self.n_aoi)          # 0-idx → 1-idx
        chosen_e = self.aoi_emb(dial_ids)                               # [B, emb_dim]

        # future signal: mean pool over F_future frames → [B, 6*C]
        if future_signal is None:
            fut_h = torch.zeros(B, self.hidden, device=device)
        else:
            fut_mean = future_signal.mean(dim=1)                        # [B, 6, C]
            fut_h    = self.future_sig_fc(
                fut_mean.reshape(B, -1))                                # [B, H]

        temporal = self.temporal_head(
            torch.cat([fused, chosen_e, fut_h], dim=-1))               # [B, 4]

        if return_attention:
            return logits, temporal, attn_map
        return logits, temporal