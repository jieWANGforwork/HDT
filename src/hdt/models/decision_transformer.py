import numpy as np
import torch
import torch.nn as nn

import transformers

from hdt.models.trajectory_model import TrajectoryModel
from hdt.models.trajectory_gpt2 import GPT2Model

class DecisionTransformerND3(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)

        self.predict_action_div = nn.Linear(hidden_size*3, self.act_dim)
        self.predict_returns_div = nn.Linear(hidden_size*3, 1)
        self.predict_action_nov = nn.Linear(hidden_size*2, self.act_dim)
        self.predict_returns_nov = torch.nn.Linear(2*hidden_size, 1)

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states_actions, sess_reward, sess_reward_user, timesteps):

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_action_embeddings = states_actions
        returns_nov_embeddings = self.embed_return_acc(sess_reward)
        #returns_nov_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)
        returns_div_embeddings = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)

        #time_embeddings = self.embed_timestep(timesteps)
        time_embeddings = timesteps


        state_action_embeddings = state_action_embeddings + time_embeddings

        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        returns_div_embeddings = returns_div_embeddings + time_embeddings

        state_return_nov = torch.cat((state_action_embeddings, returns_nov_embeddings), dim=-1)
        action_preds_nov = self.predict_action_nov(state_return_nov)  # predict next action given state
        return_nov_preds = self.predict_returns_nov(state_return_nov)

        state_return_div = torch.cat((state_action_embeddings, returns_div_embeddings, returns_nov_embeddings), dim=-1)
        action_preds_div = self.predict_action_div(state_return_div)  # predict next action given state
        return_div_preds = self.predict_returns_div(state_return_div)

        return action_preds_nov, action_preds_div, return_div_preds, return_nov_preds

class DecisionTransformerND2(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=60,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        self.predict_action_div = nn.Linear(hidden_size, self.act_dim)
        self.predict_returns_div = nn.Linear(hidden_size, 1)
        self.predict_action_nov = nn.Linear(hidden_size*2, self.act_dim)
        self.predict_returns_nov = torch.nn.Linear(2*hidden_size, 1)

        #self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions=None, sess_reward=None, sess_reward_user=None, timesteps=None, attention_mask=None, type='inter'):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to('cuda')

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        if type=='intra':
            intra=True
            state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
            action_embeddings = actions #self.embed_action(actions)
            #returns_embeddings = goal_state
            #print(returns_embeddings.shape)
            #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
            returns_nov_embeddings = self.embed_return_acc(sess_reward)
            #returns_nov_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)
            returns_div_embeddings = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)

            #returns_embeddings = (sess_reward_user + returns_embeddings_nov)/2
            #returns_div_embeddings = sess_reward_user
            #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
            #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
            #returns_div_embeddings = self.embed_return_nov(div_rewards)

            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
            returns_nov_embeddings = returns_nov_embeddings + time_embeddings
            returns_div_embeddings = returns_div_embeddings + time_embeddings


            # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # which works nice in an autoregressive sense since states predict actions
            stacked_inputs = torch.stack(
                (returns_div_embeddings, state_embeddings,  action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
            ).permute(0, 2, 1).reshape(batch_size,3*seq_length)
            #print(stacked_attention_mask.shape, attention_mask)

            # we feed in the input embeddings (not word indices as in NLP) to the model
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
                intra=intra
            )
            x = transformer_outputs['last_hidden_state']

            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
            # Rdv, S, A ->
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
            # get predictions
            state_return_div = x[:, 0]
            action_preds_div = self.predict_action_div(x[:, 1])  # predict next action given state
            return_div_preds = self.predict_returns_div(state_return_div)

            state_preds = self.predict_state(x[:,2])    # predict next state given state and action

            state_return_nov = torch.cat((x[:, 1], returns_nov_embeddings), dim=-1)
            action_preds_nov = self.predict_action_nov(state_return_nov)  # predict next action given state
            return_nov_preds = self.predict_returns_nov(state_return_nov)

            return state_preds, action_preds_nov, action_preds_div, return_div_preds, return_nov_preds
        if type=='inter':
            transformer_outputs = self.transformer(
                inputs_embeds=states,
                attention_mask=attention_mask,
                intra=False
            )
            x = transformer_outputs['last_hidden_state']
            return x

class DecisionTransformerND2_(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=60,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        self.predict_action_div = nn.Linear(hidden_size, self.act_dim)
        self.predict_returns_div = nn.Linear(hidden_size, 1)
        self.predict_action_nov = nn.Linear(hidden_size*2, self.act_dim)
        self.predict_returns_nov = torch.nn.Linear(2*hidden_size, 1)

        #self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, sess_reward_user, timesteps,
                attention_mask=None, type=None,
                intra=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_nov_embeddings = self.embed_return_acc(sess_reward)
        #returns_nov_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)
        returns_div_embeddings = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)

        #returns_embeddings = (sess_reward_user + returns_embeddings_nov)/2
        #returns_div_embeddings = sess_reward_user
        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_div_embeddings, state_embeddings,  action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,3*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            intra = intra
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # Rdv, S, A ->
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # get predictions
        state_return_div = x[:, 0]
        action_preds_div = self.predict_action_div(x[:, 1])  # predict next action given state
        return_div_preds = self.predict_returns_div(state_return_div)

        state_preds = self.predict_state(x[:,2])    # predict next state given state and action

        state_return_nov = torch.cat((x[:, 1], returns_nov_embeddings), dim=-1)
        action_preds_nov = self.predict_action_nov(state_return_nov)  # predict next action given state
        return_nov_preds = self.predict_returns_nov(state_return_nov)

        return state_preds, action_preds_nov, action_preds_div, return_div_preds, return_nov_preds

class DecisionTransformerND(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=40,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        self.predict_action_nov = nn.Linear(hidden_size*2, self.act_dim)
        self.predict_action_div = nn.Linear(hidden_size*2, self.act_dim)
        self.predict_returns_div = nn.Linear(hidden_size*2, 1)
        self.predict_returns_nov = torch.nn.Linear(2*hidden_size, 1)

        #self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, sess_reward_user, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_nov_embeddings = self.embed_return_acc(sess_reward)
        #returns_nov_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)
        returns_div_embeddings = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)

        #returns_embeddings = (sess_reward_user + returns_embeddings_nov)/2
        #returns_div_embeddings = sess_reward_user
        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings,  action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,2*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # get predictions
        #return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        #return_nov_preds = self.predict_return_nov(x[:, 0])
        state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        state_return_nov = torch.cat((x[:, 0], returns_nov_embeddings), dim=-1)
        action_preds_nov = self.predict_action_nov(state_return_nov)  # predict next action given state
        state_return_div = torch.cat((x[:, 0], returns_div_embeddings), dim=-1)
        action_preds_div = self.predict_action_div(state_return_div)  # predict next action given state
        return_div_preds = self.predict_returns_div(state_return_div)
        return_nov_preds = self.predict_returns_nov(state_return_nov)

        return state_preds, action_preds_nov, action_preds_div, return_div_preds, return_nov_preds

class DecisionTransformerUnd(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=60,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        #self.predict_action = nn.Linear(hidden_size, self.act_dim)
        self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.predict_return_nov = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, sess_reward_user, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_embeddings_nov = self.embed_return_acc(sess_reward)
        sess_reward_user = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)

        #returns_embeddings = (sess_reward_user + returns_embeddings_nov)/2
        returns_embeddings = sess_reward_user
        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        #returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        #returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings,  action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings,action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, state_embeddings, action_embeddings), dim=1 #测试accuracy为rtg的结果 bz, 10, 64->bz, 3, 10, 64
            #(returns_embeddings,  returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, returns_embeddings, state_embeddings, action_embeddings), dim=1  #位置换一下
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,3*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        return_nov_preds = self.predict_return_nov(x[:, 0])
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        #action_preds = x[:,1]
        return state_preds, action_preds, return_acc_preds, return_nov_preds

class DecisionTransformerU(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=60,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        #self.predict_action = nn.Linear(hidden_size, self.act_dim)
        self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, sess_reward_user, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        #returns_embeddings = self.embed_return_acc(sess_reward)
        sess_reward_user = sess_reward_user.unsqueeze(1).repeat(1, 20, 1)
        returns_embeddings = sess_reward_user #+ returns_embeddings

        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        #returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        #returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings,  action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings,action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, state_embeddings, action_embeddings), dim=1 #测试accuracy为rtg的结果 bz, 10, 64->bz, 3, 10, 64
            #(returns_embeddings,  returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, returns_embeddings, state_embeddings, action_embeddings), dim=1  #位置换一下
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,3*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        #return_nov_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        #action_preds = x[:,1]
        return state_preds, action_preds, return_acc_preds


class MulDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=80,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        #self.predict_action = nn.Linear(hidden_size, self.act_dim)
        self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, acc_reward, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_acc_embeddings = self.embed_return_acc(acc_reward)
        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(sess_reward)
        returns_nov_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings #+ time_embeddings
        action_embeddings = action_embeddings #+ time_embeddings
        #returns_embeddings = returns_embeddings #+ time_embeddings
        returns_acc_embeddings = returns_acc_embeddings #+ time_embeddings
        returns_nov_embeddings = returns_nov_embeddings #+ time_embeddings
        #returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            #(returns_embeddings, state_embeddings,  action_embeddings), dim=1
            (returns_acc_embeddings, returns_nov_embeddings, state_embeddings,  action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, state_embeddings, action_embeddings), dim=1 #测试accuracy为rtg的结果 bz, 10, 64->bz, 3, 10, 64
            #(returns_embeddings,  returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, returns_embeddings, state_embeddings, action_embeddings), dim=1  #位置换一下
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask,attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,4*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        return_nov_preds = self.predict_return(x[:, 1])
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,2])  # predict next action given state
        #action_preds = x[:,1]
        return state_preds, action_preds, return_acc_preds, return_nov_preds

class DecisionTransformerCat(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=20,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        self.input_state = torch.nn.Linear(hidden_size+1, self.hidden_size)

        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        #self.predict_action = nn.Linear(hidden_size, self.act_dim)
        self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_embeddings = self.embed_return_acc(sess_reward)
        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)
        #inputs = torch.cat([states, actions], dim=2)
        inputs = states #+ actions
        #print(inputs.shape)
        inputs_emb = torch.cat([sess_reward, inputs], dim=2)
        stacked_inputs = self.input_state(inputs_emb)
        time_embeddings = self.embed_timestep(timesteps)


        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = attention_mask

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        #x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_acc_preds = self.predict_return(x)  # predict next return given state and action
        #return_nov_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x)    # predict next state given state and action
        #action_preds = self.predict_action(x)  # predict next action given state
        action_preds = x
        return state_preds, action_preds, return_acc_preds

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            predict_action=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=60,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)
        self.input_state = torch.nn.Linear(hidden_size*2, self.hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        #self.predict_action = nn.Linear(hidden_size, self.act_dim)
        self.predict_action = predict_action

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, sess_reward, timesteps, attention_mask=None):
        #print(states.shape) states: 1234 actions: 2345

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states #self.input_state(states)# #states # #self.embed_state(states)
        action_embeddings = actions #self.embed_action(actions)
        #returns_embeddings = goal_state
        #print(returns_embeddings.shape)
        #returns_embeddings = returns_embeddings.unsqueeze(1).repeat(1, seq_length, 1).to(action_embeddings.device)
        returns_embeddings = self.embed_return_acc(sess_reward)
        #returns_embeddings = sess_reward.unsqueeze(1).repeat(1, 20, 1)

        #returns_acc_embeddings = self.embed_return_acc(acc_rewards)
        #returns_nov_embeddings = self.embed_return_nov(nov_rewards)
        #returns_div_embeddings = self.embed_return_nov(div_rewards)

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        #sum_return = returns_embeddings + returns_nov_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        #returns_acc_embeddings = returns_acc_embeddings + time_embeddings
        #returns_nov_embeddings = returns_nov_embeddings + time_embeddings
        #returns_div_embeddings = returns_div_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings,  action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings,action_embeddings), dim=1
            #(returns_acc_embeddings, returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, state_embeddings, action_embeddings), dim=1 #测试accuracy为rtg的结果 bz, 10, 64->bz, 3, 10, 64
            #(returns_embeddings,  returns_nov_embeddings, state_embeddings, action_embeddings), dim=1
            #(returns_nov_embeddings, returns_embeddings, state_embeddings, action_embeddings), dim=1  #位置换一下
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size,3*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # 40 leng: returns (0),returns (1), states (2), or actions (3); i.e. x[:,1,t] is the token for s_t
        # 50 leng: returns (0),returns (1),2 return,  states (3), or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        #return_nov_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        #action_preds = x[:,1]
        return state_preds, action_preds, return_acc_preds


class UserDecision(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)


    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=3,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=3,
            #n_layer=1,
            #n_head=1,
            **kwargs
        )
        #print(config)
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)

        #balance 3rewards
        #self.reward_gate = XXX

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return_acc = torch.nn.Linear(1, hidden_size)
        self.embed_return_nov = torch.nn.Linear(1, hidden_size)
        self.embed_return_div = torch.nn.Linear(1, hidden_size)
        '''self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)'''

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.hidden_size)
        #self.predict_action = nn.Sequential(
         #   *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        #)
        self.predict_action = nn.Linear(hidden_size, self.hidden_size)

        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, user_reward, acc_rewards = None,nov_rewards= None, div_rewards= None, returns_to_go = None, timesteps = None, attention_mask=None):
        '''    #S,R, A -> R, A , S':  goal是A或者S'
        '''
        step_length = 3
        batch_size, seq_length = states.shape[0], 1

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head:先将state，action和return的维度转换，值是直接的数据
        state_embeddings = states.unsqueeze(1) #self.embed_state(states)
        action_embeddings = actions.unsqueeze(1) #self.embed_action(actions)
        returns_embeddings = self.embed_return_acc(user_reward).unsqueeze(1)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, returns_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, step_length*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print(stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1 #bz,3,lens
        ).permute(0, 2, 1).reshape(batch_size, step_length*seq_length)
        #print(stacked_attention_mask.shape, attention_mask)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # return, action, state
        x = x.reshape(batch_size, seq_length, step_length, self.hidden_size).permute(0, 2, 1, 3)
        #print(x.shape)
        # get predictions R, A , S'
        return_acc_preds = self.predict_return(x[:,0])  # predict next return given state and action
        #return_nov_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        #print(state_preds.shape)
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_acc_preds

    def get_actionget_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
