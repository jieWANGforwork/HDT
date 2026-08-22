from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from hdt.models.attention import MultiHeadAttention, MultiHeadAttentionCross, PositionwiseFeedForward
from hdt.models.decision_transformer import DecisionTransformer, DecisionTransformerND2
from hdt.utils import extract_axis_1


@dataclass
class TransformerVariant:
    embed_dim: int = 100
    n_layer: int = 1
    n_head: int = 1
    activation_function: str = "relu"
    dropout: float = 0.1

    def as_kwargs(self) -> Dict[str, object]:
        return {
            "embed_dim": self.embed_dim,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
        }


class HDT(nn.Module):
    """Hierarchical Transformer for multi-objective sequential recommendation.

    Modes map to the original project entry points:
    - inter_only: DUTST.py / DUTST_div.py
    - single_objective: DUTDT.py / DUTDT_div_return.py
    - multi_objective: DUTDT_ND.py
    """

    def __init__(
        self,
        hidden_size: int,
        item_num: int,
        state_size: int,
        dropout: float,
        discount: float,
        device: torch.device,
        mode: str = "multi_objective",
        variant: Optional[TransformerVariant] = None,
        num_heads: int = 1,
        action_embedding_source: str = "actions",
    ):
        super().__init__()
        if mode not in {"inter_only", "single_objective", "multi_objective"}:
            raise ValueError(f"Unsupported HDT mode: {mode}")
        if action_embedding_source not in {"states", "actions"}:
            raise ValueError("action_embedding_source must be 'states' or 'actions'")

        self.mode = mode
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.discount = discount
        self.device = device
        self.action_embedding_source = action_embedding_source

        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        nn.init.normal_(self.positional_embeddings.weight, 0, 0.01)

        self.in_ff = nn.Linear(hidden_size * 2, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        self.s_fc_user = nn.Linear(hidden_size, item_num)

        self.embed_user_reward = nn.Linear(1, hidden_size)
        self.predict_user_reward = nn.Linear(hidden_size, 1)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.user_dt = MultiHeadAttentionCross(hidden_size, hidden_size, num_heads, 0)
        self.return_embeddings = nn.Embedding(state_size, hidden_size)

        variant = variant or TransformerVariant(embed_dim=hidden_size, n_head=num_heads, dropout=dropout)
        variant_kwargs = variant.as_kwargs()

        if mode == "single_objective":
            self.session_dt = DecisionTransformer(
                state_dim=1,
                act_dim=item_num,
                max_length=state_size,
                max_ep_len=state_size,
                hidden_size=hidden_size,
                n_inner=4 * int(variant_kwargs["embed_dim"]),
                predict_action=self.s_fc,
                **variant_kwargs,
            )
        elif mode == "multi_objective":
            self.session_dt = DecisionTransformerND2(
                state_dim=1,
                act_dim=item_num,
                max_length=state_size,
                max_ep_len=state_size,
                hidden_size=hidden_size,
                n_inner=4 * int(variant_kwargs["embed_dim"]),
                predict_action=self.s_fc,
                **variant_kwargs,
            )
        else:
            self.session_dt = None

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=self.device)

    def mask_zeros(self, repr_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(repr_tensor, device=self.device)

    def _user_state(
        self,
        user_reward: torch.Tensor,
        action_state: torch.Tensor,
    ) -> torch.Tensor:
        user_reward = self.embed_user_reward(user_reward).unsqueeze(1)
        user_stacked_inputs = torch.cat((user_reward, action_state), dim=1)
        user_stacked_inputs = self.embed_ln(user_stacked_inputs)
        if self.mode == "multi_objective":
            attention_mask = torch.ones(
                user_stacked_inputs.shape[:2],
                dtype=torch.long,
                device=user_stacked_inputs.device,
            )
            return self.session_dt.forward(user_stacked_inputs, attention_mask=attention_mask, type="inter")
        return self.user_dt(user_stacked_inputs, user_stacked_inputs)

    def _sequence_state(self, states: torch.Tensor, user_state: torch.Tensor, res_q: Optional[torch.Tensor] = None):
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        repeated_user_state = user_state.unsqueeze(1).repeat(1, self.state_size, 1)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size, device=self.device))
        seq = self.emb_dropout(inputs_emb)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(
            repeated_user_state,
            seq - 0.1 * repeated_user_state,
            seq_normalized,
            res_q=res_q,
        )
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        action_preds0 = self.s_fc(ff_out)
        len_states = mask.sum(dim=1).squeeze(1).long()
        state_hidden = extract_axis_1(ff_out, len_states - 1).squeeze(1)
        attention_mask = mask.squeeze(-1)
        return ff_out, action_preds0, state_hidden, attention_mask

    def _align_to_states(self, values: Optional[torch.Tensor], states: torch.Tensor) -> Optional[torch.Tensor]:
        if values is None or values.dim() < 2:
            return values
        state_len = states.size(1)
        if values.size(1) == state_len + 1:
            return values[:, 1:]
        if values.size(1) > state_len:
            return values[:, :state_len]
        return values

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        user_state: torch.Tensor,
        action_state: torch.Tensor,
        user_mask: torch.Tensor,
        user_reward: torch.Tensor,
        poss_reward: Optional[torch.Tensor] = None,
        user_reward_nov: Optional[torch.Tensor] = None,
        poss_reward_nov: Optional[torch.Tensor] = None,
    ):
        actions = self._align_to_states(actions, states)
        poss_reward = self._align_to_states(poss_reward, states)
        poss_reward_nov = self._align_to_states(poss_reward_nov, states)
        user_state_tokens = self._user_state(user_reward, action_state)
        user_state_vec = user_state_tokens[:, -1]
        user_preds = self.s_fc(user_state_vec)

        res_q = action_state if self.mode == "multi_objective" and action_state.size(1) == self.state_size else None
        ff_out, action_preds0, state_hidden, attention_mask = self._sequence_state(states, user_state_vec, res_q=res_q)
        action_target = torch.clone(actions)
        action_preds0 = action_preds0.reshape(-1, self.item_num)[attention_mask.reshape(-1) > 0]
        action_target_masked = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        if self.mode == "inter_only":
            user_reward_preds = self.predict_user_reward(user_state_tokens[:, -1])
            return action_preds0, action_target_masked, user_state_tokens[:, -1], state_hidden, user_reward_preds, user_preds

        timesteps = torch.arange(self.state_size, device=self.device)
        if self.action_embedding_source == "actions":
            action_embeddings = self.item_embeddings(actions.long())
        else:
            action_embeddings = self.item_embeddings(states.long())

        if self.mode == "single_objective":
            if poss_reward is None:
                raise ValueError("single_objective mode requires poss_reward")
            poss_reward = poss_reward.unsqueeze(2)
            state_preds, action_preds, reward_preds = self.session_dt.forward(
                ff_out,
                action_embeddings,
                poss_reward,
                timesteps,
                attention_mask=attention_mask,
            )
            action_preds = action_preds.reshape(-1, self.item_num)[attention_mask.reshape(-1) > 0]
            poss_reward_preds = reward_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
            poss_reward_target = poss_reward.reshape(-1)[attention_mask.reshape(-1) > 0]
            return (
                user_preds,
                action_preds0,
                action_preds,
                action_target_masked,
                state_hidden,
                poss_reward_preds,
                poss_reward_target,
            )

        if poss_reward is None or poss_reward_nov is None:
            raise ValueError("multi_objective mode requires poss_reward and poss_reward_nov")
        poss_reward = poss_reward.unsqueeze(2)
        poss_reward_nov = poss_reward_nov.unsqueeze(2)
        state_preds, action_preds_nov, action_preds_div, return_div_preds, return_nov_preds = self.session_dt.forward(
            ff_out,
            action_embeddings,
            poss_reward_nov,
            user_state_tokens[:, 0],
            timesteps,
            attention_mask=attention_mask,
            type="intra",
        )
        action_preds_nov = action_preds_nov.reshape(-1, self.item_num)[attention_mask.reshape(-1) > 0]
        action_preds_div = action_preds_div.reshape(-1, self.item_num)[attention_mask.reshape(-1) > 0]
        poss_reward_preds = return_div_preds.reshape(-1)[attention_mask.reshape(-1) > 0]
        poss_reward_target = poss_reward.reshape(-1)[attention_mask.reshape(-1) > 0]
        return (
            user_preds,
            action_preds0,
            action_preds_nov,
            action_preds_div,
            action_target_masked,
            state_hidden,
            poss_reward_preds,
            poss_reward_target,
        )
