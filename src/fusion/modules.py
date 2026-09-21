"""
Fusion modules for audio + EGG embedding fusion.

All modules take (e_audio, e_egg) of shape (B, D) each and return a fused
embedding of shape (B, D). Caller is responsible for L2-normalization at
scoring time.

Backs paper Table 1 (fusion systems + ablations) and provides the gate that the
three interpretability controls analyse.

The system reported in the paper is the untied, NO-TRUNK, per-dimension gate,
reached through the factory as `no_trunk_vector_gate`. The three ablations are
`vector_gate` (+trunk), `no_trunk_tied_vector_gate` (tied) and
`no_trunk_scalar_gate` (scalar); `mlp` is the learned-fusion baseline.

Trainable parameter counts at D_a = D_e = D_out = 192, hidden = 384, as
emitted by src/figures/table1_main_results.py:
    mlp                        222,528
    vector_gate (+trunk)       221,952
    no_trunk_tied_vector_gate   73,920
    no_trunk_scalar_gate           770
    no_trunk_vector_gate       147,840   <- primary

The primary gate reaches its result with roughly two thirds of the MLP
baseline's parameters; the comparison is not parameter-matched.
"""

import torch
import torch.nn as nn


def get_fusion(name: str, audio_dim: int, egg_dim: int,
               hidden_dim: int, output_dim: int, dropout: float, gate_input_norm: str = "none") -> nn.Module:
    """Factory: return the requested fusion module."""
    name = name.lower()

    if name == "mlp":
        return MLPFusion(audio_dim, egg_dim, hidden_dim, output_dim, dropout)
    if name == "scalar_gate":
        return ScalarGateFusion(audio_dim, egg_dim, dropout)
    if name =="untied_scalar_gate":
        return UntiedScalarGateFusion(audio_dim, egg_dim, dropout)
    if name == "vector_gate":
        return VectorGateFusion(audio_dim, egg_dim, hidden_dim, output_dim, dropout)
    if name == "tied_vector_gate":
        return TiedVectorGateFusion(audio_dim, egg_dim, hidden_dim, output_dim, dropout)
    if name == "no_trunk_vector_gate":
        return NoTrunkVectorGateNormProbeFusion(audio_dim, egg_dim, hidden_dim, output_dim, dropout, gate_input_norm)
    if name == "no_trunk_scalar_gate":
        return NoTrunkUntiedScalarGateFusion(audio_dim, egg_dim, dropout)
    if name == "no_trunk_tied_vector_gate":
        return NoTrunkTiedVectorGateFusion(audio_dim, egg_dim, hidden_dim, output_dim, dropout)
    
    raise ValueError(f"Unknown fusion type: {name}")


class MLPFusion(nn.Module):
    """Concat -> Linear -> LN -> ReLU -> Dropout -> Linear. No gating."""
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(audio_dim + egg_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, e_audio, e_egg):
        x = torch.cat([e_audio, e_egg], dim=-1)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_gate(self, e_audio, e_egg):
        """No gate in MLP fusion. Returns None for API compatibility."""
        return None


class ScalarGateFusion(nn.Module):
    """
    Tied scalar gate (ablation).
        g = sigmoid(MLP([e_a; e_e]))    in (0, 1)
        fused = g * e_audio + (1 - g) * e_egg

    Tied: only one gate; the two modalities trade off zero-sum.
    Output dim == audio_dim == egg_dim. No projection layer.
    """
    def __init__(self, audio_dim, egg_dim, dropout):
        super().__init__()
        assert audio_dim == egg_dim, "Scalar gate requires audio_dim == egg_dim"
        
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )
        
    def forward(self, e_audio, e_egg):
        g = torch.sigmoid(self.gate(torch.cat([e_audio, e_egg], dim=-1)))  # (B, 1)
        return g * e_audio + (1 - g) * e_egg

    def get_gate(self, e_audio, e_egg):
        """Return scalar gate value(s) for analysis. Shape: (B, 1)."""
        with torch.no_grad():
            g = torch.sigmoid(self.gate(torch.cat([e_audio, e_egg], dim=-1)))
        return g

class UntiedScalarGateFusion(nn.Module):
    """
    Untied scalar gate (ablation).
        [g_a, g_e] = sigmoid(MLP([e_a; e_e]))    each in (0, 1), independent
        fused = g_a * e_audio + g_e * e_egg

    Untied: one scalar per modality, NOT zero-sum. The model can up- or
    down-weight each modality freely (e.g. suppress both, or keep both).
    Single-factor change from ScalarGateFusion: vector -> scalar is the only
    difference vs the primary; here tied -> untied is the only difference vs
    ScalarGateFusion.
    Output dim == audio_dim == egg_dim. No projection layer.
    """
    def __init__(self, audio_dim, egg_dim, dropout):
        super().__init__()
        assert audio_dim == egg_dim, "Scalar gate requires audio_dim == egg_dim"

        self.gate = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 2),          # was 1 (tied) -> 2 (untied)
        )

    def forward(self, e_audio, e_egg):
        g = torch.sigmoid(self.gate(torch.cat([e_audio, e_egg], dim=-1)))  # (B, 2)
        g_a = g[:, 0:1]                 # (B, 1)
        g_e = g[:, 1:2]                 # (B, 1)
        return g_a * e_audio + g_e * e_egg

    def get_gate(self, e_audio, e_egg):
        """
        Return per-modality scalar gates for analysis. Shape: (B, 2),
        column 0 = g_audio, column 1 = g_egg.
        """
        with torch.no_grad():
            g = torch.sigmoid(self.gate(torch.cat([e_audio, e_egg], dim=-1)))
        return g

class VectorGateFusion(nn.Module):
    """
    Untied per-dimension gates (primary contribution).
        trunk = LN(ReLU(Linear([e_a; e_e])))
        g_a = sigmoid(head_a(trunk))    in (0, 1)^D
        g_e = sigmoid(head_e(trunk))    in (0, 1)^D
        fused = g_a * e_audio + g_e * e_egg

    Untied: each modality has its own gate. The two can both shrink
    (uncertain) or both saturate (both reliable) — no zero-sum constraint.

    Justification for untied vs tied:
        - At -10dB cafeteria + clean EGG, audio is destroyed and EGG is
          reliable. Tied gate forces g_egg = 1 - g_audio, which works here.
        - But at clean audio + clean EGG, tied still forces a tradeoff
          even though both are informative. Untied lets both saturate.
        - Discuss this in Ch 3.4.3.
    """
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        assert audio_dim == egg_dim == output_dim, \
            "Vector gate requires audio_dim == egg_dim == output_dim"
        self.trunk_audio = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.trunk_egg = nn.Sequential(
            nn.Linear(egg_dim, output_dim),
            nn.Dropout(dropout),
            nn.Tanh()
        )
        self.head_audio_gate = nn.Linear(audio_dim + egg_dim, output_dim)
        self.head_egg_gate = nn.Linear(audio_dim + egg_dim, output_dim)

    def forward(self, e_audio, e_egg):
        concat_emb = torch.cat([e_audio,e_egg], dim=-1)
        e_audio_trunked = self.trunk_audio(e_audio)
        e_egg_trunked = self.trunk_egg(e_egg)
        g_audio = torch.sigmoid(self.head_audio_gate(concat_emb))
        g_egg = torch.sigmoid(self.head_egg_gate(concat_emb))
        return g_audio * e_audio_trunked + g_egg * e_egg_trunked

    def get_gate(self, e_audio, e_egg):
        """
        Return both gate vectors for analysis.
        Shape: dict with 'g_audio' and 'g_egg', each (B, D).
        """
        with torch.no_grad():
            concat_emb = torch.cat([e_audio,e_egg], dim=-1)
            g_audio = torch.sigmoid(self.head_audio_gate(concat_emb))
            g_egg = torch.sigmoid(self.head_egg_gate(concat_emb))
        return {'g_audio': g_audio, 'g_egg': g_egg}


class TiedVectorGateFusion(nn.Module):
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, e_audio, e_egg):
        h = self.trunk(torch.cat([e_audio, e_egg], dim=-1))
        z = torch.sigmoid(self.head(h))
        return z * e_audio + (1 - z) * e_egg

    def get_gate(self, e_audio, e_egg):
        with torch.no_grad():
            h = self.trunk(torch.cat([e_audio, e_egg], dim=-1))
            z = torch.sigmoid(self.head(h))
        return {'z': z}
    
class NoTrunkTiedVectorGateFusion(nn.Module):
    """
    Tied per-dimension vector gate, NO trunk (ablation: removes 'untied' from
    the no-trunk vector primary). One gate vector, zero-sum per dimension.
        z     = sigmoid( Dropout( Linear([e_a; e_e]) ) )   in (0,1)^D
        fused = z * e_audio + (1 - z) * e_egg
    Identical to the no-trunk vector primary except: one head + (1 - z),
    instead of two independent heads. That single difference (untied -> tied)
    is what this ablation isolates.
    """
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.head = nn.Sequential(
        nn.Linear(audio_dim + egg_dim, output_dim),
        nn.Dropout(dropout)
        )

    def forward(self, e_audio, e_egg):
        z = torch.sigmoid(self.head(torch.cat([e_audio, e_egg], dim=-1)))
        return z * e_audio + (1 - z) * e_egg

    def get_gate(self, e_audio, e_egg):
        with torch.no_grad():
            z = torch.sigmoid(self.head(torch.cat([e_audio, e_egg], dim=-1)))
        return {'z': z}


class NoTrunkUntiedScalarGateFusion(nn.Module):
    """
    Untied scalar gate, NO trunk (ablation: removes 'per-dimension' from the
    no-trunk vector primary). Single linear map per modality, output dim 1.
        g_a = sigmoid( Dropout( Linear([e_a; e_e]) ) )   in (0,1)
        g_e = sigmoid( Dropout( Linear([e_a; e_e]) ) )   in (0,1)
        fused = g_a * e_audio + g_e * e_egg

    Identical to NoTrunkVectorGate*Fusion except the head output dim is 1
    instead of D. That single difference (vector -> scalar) is what this
    ablation isolates. No trunk, no ReLU, dropout after the linear (matching
    the primary), two independent heads.
    """
    def __init__(self, audio_dim, egg_dim, dropout):
        super().__init__()
        assert audio_dim == egg_dim, "Scalar gate requires audio_dim == egg_dim"
        self.head_audio = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, 1),
            #nn.Dropout(dropout),
        )
        self.head_egg = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, 1),
            #nn.Dropout(dropout),
        )

    def forward(self, e_audio, e_egg):
        x = torch.cat([e_audio, e_egg], dim=-1)
        g_a = torch.sigmoid(self.head_audio(x))   # (B, 1)
        g_e = torch.sigmoid(self.head_egg(x))     # (B, 1)
        return g_a * e_audio + g_e * e_egg

    def get_gate(self, e_audio, e_egg):
        """Per-modality scalar gates. Dict to match the vector gate's interface."""
        with torch.no_grad():
            x = torch.cat([e_audio, e_egg], dim=-1)
            g_a = torch.sigmoid(self.head_audio(x))
            g_e = torch.sigmoid(self.head_egg(x))
        return {'g_audio': g_a, 'g_egg': g_e}     # each (B, 1)

 

class NoTrunkVectorGateFusion(nn.Module):
    """Untied per-dimension gates, no trunk (paper-style single linear gate map).

    g_audio = sigmoid(W_a · [e_a; e_egg])
    g_egg   = sigmoid(W_e · [e_a; e_egg])
    fused   = g_audio ⊙ e_audio + g_egg ⊙ e_egg

    Ablation: tests whether the trunk + ReLU in VectorGateFusion adds value
    over a direct linear gate map (closer to Jing et al. 2023 Eq. 3).
    """
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        assert audio_dim == egg_dim == output_dim
        self.head_audio = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.head_egg = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, output_dim),
            nn.Dropout(dropout),
        )
        # hidden_dim unused — kept in signature for factory compatibility

    def forward(self, e_audio, e_egg):
        x = torch.cat([e_audio, e_egg], dim=-1)
        g_audio = torch.sigmoid(self.head_audio(x))
        g_egg   = torch.sigmoid(self.head_egg(x))
        return g_audio * e_audio + g_egg * e_egg

    def get_gate(self, e_audio, e_egg):
        with torch.no_grad():
            x = torch.cat([e_audio, e_egg], dim=-1)
            g_audio = torch.sigmoid(self.head_audio(x))
            g_egg   = torch.sigmoid(self.head_egg(x))
        return {'g_audio': g_audio, 'g_egg': g_egg}


class NoTrunkVectorGateNormProbeFusion(nn.Module):
    """
    No-trunk untied vector gate with a switchable normalization on the
    GATE INPUT ONLY. Used to test whether the gate's SNR-tracking is driven
    by reading embedding magnitude, or by embedding direction/content.

        x_gate = norm([e_a; e_e])            # per switch; gate input only
        g_a    = sigmoid(W_a · x_gate)       in (0,1)^D
        g_e    = sigmoid(W_e · x_gate)       in (0,1)^D
        fused  = g_a ⊙ e_audio + g_e ⊙ e_egg # FUSION USES ORIGINAL EMBEDDINGS

    gate_input_norm:
        'none'      -> reproduces the primary (gate sees raw, unnormalized embeds)
        'l2'        -> per-sample L2: gate sees unit-norm embeds (magnitude removed)
        'layernorm' -> per-sample LayerNorm, no affine (mean/scale removed)

    Only the gate's INPUT is normalized; the fused output is built from the
    original embeddings, so this isolates "what the gate decides on" without
    redesigning the fused representation. Retrain per setting; train and eval
    must use the same setting.
    """
    def __init__(self, audio_dim, egg_dim, hidden_dim, output_dim, dropout,
                 gate_input_norm="none", eps=1e-6):
        super().__init__()
        assert audio_dim == egg_dim == output_dim
        assert gate_input_norm in ("none", "l2", "layernorm")
        self.gate_input_norm = gate_input_norm
        self.eps = eps

        self.head_audio = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.head_egg = nn.Sequential(
            nn.Linear(audio_dim + egg_dim, output_dim),
            nn.Dropout(dropout),
        )
        if gate_input_norm == "layernorm":
            # no affine: a learned scale could re-encode magnitude, defeating the test
            self.ln = nn.LayerNorm(audio_dim, elementwise_affine=False)
        # hidden_dim unused — kept in signature for factory compatibility

    def _norm_for_gate(self, e):
        if self.gate_input_norm == "none":
            return e
        if self.gate_input_norm == "l2":
            return e / e.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return self.ln(e)  # layernorm

    def _gate_input(self, e_audio, e_egg):
        return torch.cat([self._norm_for_gate(e_audio),
                          self._norm_for_gate(e_egg)], dim=-1)

    def forward(self, e_audio, e_egg):
        x = self._gate_input(e_audio, e_egg)
        g_audio = torch.sigmoid(self.head_audio(x))
        g_egg   = torch.sigmoid(self.head_egg(x))
        return g_audio * e_audio + g_egg * e_egg   # original embeddings

    def get_gate(self, e_audio, e_egg):
        with torch.no_grad():
            x = self._gate_input(e_audio, e_egg)
            g_audio = torch.sigmoid(self.head_audio(x))
            g_egg   = torch.sigmoid(self.head_egg(x))
        return {'g_audio': g_audio, 'g_egg': g_egg}