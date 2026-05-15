import torch
import torch.nn as nn
import torch.nn.functional as F

from molecule_design import MoleculeDesign

from xlstm.xlstm_large.model import xLSTMLargeBlockStack, xLSTMLargeConfig


class SparseGraphGNN_mLSTM(nn.Module):
    
    def __init__(
        self,
        config,
        d_model: int,
        num_gnn_layers: int,
        gnn_dropout: float,
        gnn_use_jk: bool,
        num_lstm_layers: int,
        num_lstm_heads: int,
        mode: str ="train",  # "train", "inference"
    ):
        super().__init__()

        self.device = torch.device(config.training_device)
        
        num_possible_atom_types = len(config.atom_vocabulary) + 1

        atom_padding_idx = num_possible_atom_types

        num_possible_bonds = MoleculeDesign.maximum_bond_order

        max_valence = max(config.atom_vocabulary[atom]["valence"] for atom in config.atom_vocabulary)
        
        degree_padding_idx = max_valence + 1

        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1

        bond_vocab_size = MoleculeDesign.virtual_bond_idx + 2

        # Encoder (GNN)
        self.encoder = SparseGNNEncoder(
        num_possible_atom_types=num_possible_atom_types,
        atom_padding_idx=atom_padding_idx,
        degree_padding_idx=degree_padding_idx,
        bond_padding_idx=bond_padding_idx,
        max_valence=max_valence,
        bond_vocab_size=bond_vocab_size,
        d_model=d_model,
        num_layers=num_gnn_layers,
        dropout=gnn_dropout,
        use_jk=gnn_use_jk,
        )

        # Sequence model (mLSTM)
        self.sequence_model = xLSTMSequenceModel(
            config=config,
            d_model=d_model,
            num_layers=num_lstm_layers,
            num_heads=num_lstm_heads,
            mode=mode,
        )

        # Action head (STRICT Transformer contract)


        self.action_head = GraphixActionHead(
            d_model=d_model,
            num_possible_atom_types=num_possible_atom_types,
            num_possible_bonds=num_possible_bonds,
        )

        self.to(self.device)

    def forward(self, input_data):
        """
        input_data = batch["input"]
        """

        # (B, N, D), (B, N)
        h, padding_mask = self.encoder(input_data)

        # (B, N, D)
        h, _ = self.sequence_model(h)

        # logits:
        #   level0: (B, N_atoms + virtual)
        #   level1: (B, N_atoms)
        #   level2: (B, num_bond_types)
        level0_logits, level1_logits, level2_logits = self.action_head(h, padding_mask)

        return level0_logits, level1_logits, level2_logits
    
    def get_weights(self):
        return dict_to_cpu(self.state_dict())
    

class SparseGNNLayer(nn.Module):
    """
    Sparse message-passing layer.
    Drop-in replacement for DenseGNNLayer.
    """
    def __init__(
        self,
        d_model: int,
        bond_vocab_size: int,
        bond_padding_idx: int,
        dropout: float = 0.0,
        no_bond_idx: int = 0,
    ):
        super().__init__()
        self.dropout = dropout
        self.no_bond_idx = no_bond_idx
        self.bond_padding_idx = bond_padding_idx

        self.bond_emb = nn.Embedding(
            bond_vocab_size,
            d_model,
            padding_idx=bond_padding_idx,
        )

        self.lin_msg = nn.Linear(d_model, d_model)
        self.lin_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,          # (B, N, D)
        bonds: torch.Tensor,      # (B, N, N)
        node_valid: torch.Tensor  # (B, N) bool
    ) -> torch.Tensor:

        B, N, D = h.shape

        # valid edges: real bond + both nodes valid
        edge_exist = (bonds != self.no_bond_idx) & (bonds != self.bond_padding_idx)
        node_pair_valid = node_valid[:, :, None] & node_valid[:, None, :]
        edge_valid = edge_exist & node_pair_valid

        # indices of edges: j -> i
        b_idx, i_idx, j_idx = edge_valid.nonzero(as_tuple=True)

        if b_idx.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            h_j = h[b_idx, j_idx]                    # (E, D) one string correspondes one edge - the strings can repeat
            bond_types = bonds[b_idx, i_idx, j_idx]  # (E,)
            e_ij = self.bond_emb(bond_types)         # (E, D)

            msg = F.relu(self.lin_msg(h_j + e_ij))   # (E, D)

            # scatter add into destination nodes
            flat_dst = b_idx * N + i_idx             # (E,) indices of destination nodes repeated according to their incoming edges
            agg_flat = h.new_zeros((B * N, D))
            agg_flat.index_add_(0, flat_dst, msg)
            agg = agg_flat.view(B, N, D)

        h = self.norm(h + F.dropout(agg, self.dropout, self.training))
        out = self.lin_out(F.relu(h))
        out = self.norm(h + F.dropout(out, self.dropout, self.training))
        return out
    

class SparseGNNEncoder(nn.Module):
    """
    Sparse GNN + Jumping Knowledge.
    Drop-in replacement for DenseGNNEncoder.
    """
    def __init__(
        self,
        num_possible_atom_types: int,
        atom_padding_idx: int,
        degree_padding_idx: int,
        bond_padding_idx: int,
        max_valence: int,
        bond_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_jk: bool = True,
    ):
        super().__init__()

        self.use_jk = use_jk
        self.atom_padding_idx = atom_padding_idx

        # node embeddings
        self.atom_emb = nn.Embedding(
            num_possible_atom_types + 1,
            d_model,
            padding_idx=atom_padding_idx,
        )
        self.degree_emb = nn.Embedding(
            max_valence + 2,
            d_model,
            padding_idx=degree_padding_idx,
        )

        # task invariants
        self.picked_atom_emb = nn.Embedding(3, d_model, padding_idx=0)
        self.virtual_level_emb = nn.Embedding(3, d_model)

        # sparse GNN layers
        self.layers = nn.ModuleList([
            SparseGNNLayer(
                d_model=d_model,
                bond_vocab_size=bond_vocab_size,
                bond_padding_idx=bond_padding_idx,
                dropout=dropout,
                no_bond_idx=0,
            )
            for _ in range(num_layers)
        ])

        # Jumping Knowledge
        if self.use_jk:
            self.jk_proj = nn.Linear(num_layers * d_model, d_model)

    def forward(self, x: dict):
        atoms = x["atoms"]                  # (B, N)
        degree = x["atoms_degree"]          # (B, N)
        bonds = x["bonds"]                  # (B, N, N)
        picked = x["picked_atom_mhe"]       # (B, N)
        level_idx = x["level_idx"]          # (B,)

        node_valid = atoms != self.atom_padding_idx

        # initial node features
        h = self.atom_emb(atoms)
        if h.size(1) > 1:
            h[:, 1:] = h[:, 1:] + self.degree_emb(degree[:, 1:])
        h = h + self.picked_atom_emb(picked)
        h[:, 0] = h[:, 0] + self.virtual_level_emb(level_idx)

        # message passing
        h_layers = []
        for layer in self.layers:
            h = layer(h, bonds, node_valid)
            h_layers.append(h)

        # Jumping Knowledge
        if self.use_jk:
            h = torch.cat(h_layers, dim=-1)
            h = self.jk_proj(h)
        else:
            h = h_layers[-1]

        return h, node_valid
    

class xLSTMSequenceModel(nn.Module):
    
    
        def __init__(self, config, d_model, num_layers, num_heads, mode):
            super().__init__()

            xlstm_cfg = xLSTMLargeConfig(
                embedding_dim=d_model,
                num_heads=num_heads,
                num_blocks=num_layers,
                vocab_size=1,  # dummy
                mode=mode, 
                chunkwise_kernel="chunkwise--native_autograd", #chunkwise_kernel="chunkwise--triton_xl_chunk" native_autograd
                sequence_kernel="native_sequence__native",  #sequence_kernel="native_sequence__triton"
                step_kernel="native",  #step_kernel="triton"
            )

            self.backbone = xLSTMLargeBlockStack(xlstm_cfg)


   

        def forward(self, x, state=None):
            h, state = self.backbone(x, state)
            return h, state
    
class GraphixActionHead(nn.Module):
    def __init__(self, d_model, num_possible_atom_types, num_possible_bonds):
        super().__init__()
        
        self.num_possible_bonds = num_possible_bonds

        self.virtual_atom_linear = nn.Linear(
            d_model,
            num_possible_atom_types + num_possible_bonds
        )

        # per-atom logits: level 0 and level 1
        self.bond_atom_linear = nn.Linear(d_model, 2)

    def forward(self, h, padding_mask):
        """
        h: (B, N, D)
        padding_mask: (B, N)
        """
        virtual = h[:, 0]      # (B, D)
        atoms   = h[:, 1:]     # (B, N-1, D)

        # virtual atom
        virtual_atom_mapping = self.virtual_atom_linear(virtual)
        level0_virtual = virtual_atom_mapping[:, :-self.num_possible_bonds]   # (B, num_possible_atom_types: vocab + virtual)
        level2_logits  = virtual_atom_mapping[:, -self.num_possible_bonds:]   # (B, num_possible_bonds)

        # atoms
        atom_logits = self.bond_atom_linear(atoms)          # (B, N-1, 2)
        level0_atoms = atom_logits[..., 0]                  # (B, N-1)
        level1_logits = atom_logits[..., 1]                 # (B, N-1)

        # padding
        level0_atoms = level0_atoms.masked_fill(
            ~padding_mask[:, 1:], -torch.inf
        )
        level1_logits = level1_logits.masked_fill(
            ~padding_mask[:, 1:], -torch.inf
        )

        # final concat
        level0_logits = torch.cat(
            [level0_virtual, level0_atoms],
            dim=1
        )

        return level0_logits, level1_logits, level2_logits
    
def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict