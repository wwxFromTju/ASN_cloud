import json

import torch as th
import torch.nn as nn
import torch.nn.functional as F

def read_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    return data


class AsnHyperAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AsnHyperAgent, self).__init__()
        self.args = args

        print(args)

        self.map_name = args.env_args['map_name'] + '_obs'
        map_config = read_json('./obs_config.json')

        assert self.map_name in map_config.keys(), 'map config not find'
        assert input_shape == map_config[self.map_name]['model_input_size'], 'input shape mismatch'

        self.enemies_feat_start = map_config[self.map_name]['model_input_compose']['0']['size']
        self.enemies_num, self.enemy_feats_size = map_config[self.map_name]['model_input_compose']['1']['size']

        # network struct
        self.env_info_fc1 = nn.Linear(input_shape, args.asn_hidden_size)
        self.env_info_fc2 = nn.Linear(args.asn_hidden_size, args.asn_hidden_size)
        self.env_info_rnn3 = nn.GRUCell(args.asn_hidden_size, args.asn_hidden_size)

        # no-op + stop + up, down, left, right
        self.wo_action_fc = nn.Linear(args.asn_hidden_size, 6)
        
        self.enemies_info_fc1 = nn.Linear(self.enemy_feats_size, args.asn_hidden_size)
        self.enemies_info_fc2 = nn.Linear(args.asn_hidden_size, args.asn_hidden_size)
        self.enemies_info_rnn3 = nn.GRUCell(args.asn_hidden_size, args.asn_hidden_size)

        self.hyper_w_and_b_attack_actions = nn.Sequential(
            nn.Linear(self.enemy_feats_size, args.asn_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.asn_hidden_size, args.asn_hidden_size * 1 + 1)
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.env_info_fc1.weight.new(1, self.args.asn_hidden_size * (1 + self.enemies_num)).zero_()

    def forward(self, inputs, hidden_state):
        # print(inputs.shape)
        # print(hidden_state.shape)

        enemies_feats = [inputs[:, self.enemies_feat_start + i * self.enemy_feats_size: self.enemies_feat_start + self.enemy_feats_size * (1 + i)] for i in range(self.enemies_num)]


        h_in = th.split(hidden_state, self.args.asn_hidden_size, dim=-1)
        h_in_env = h_in[0].reshape(-1, self.args.asn_hidden_size)
        h_in_enemies = [_h.reshape(-1, self.args.asn_hidden_size) for _h in h_in[1:]]

        env_hidden_1 = F.relu(self.env_info_fc1(inputs))
        env_hidden_2 = self.env_info_fc2(env_hidden_1)
        h_env = self.env_info_rnn3(env_hidden_2, h_in_env)

        wo_action_fc_Q = self.wo_action_fc(h_env)

        enemies_hiddent_1 = [F.relu(self.enemies_info_fc1(enemy_info)) for enemy_info in enemies_feats]
        enemies_hiddent_2 = [self.enemies_info_fc2(enemy_info) for enemy_info in enemies_hiddent_1]
        enemies_h_hiddent_3 = [self.enemies_info_rnn3(enemy_info, enemy_h) for enemy_info, enemy_h in zip(enemies_hiddent_2, h_in_enemies)]


        w_and_b_attack = [self.hyper_w_and_b_attack_actions(enemy_info) for enemy_info in enemies_feats]
        attack_enemy_id_Q = [th.sum(h_env * enemy_info * param[..., :-1] + param[-1], dim=-1, keepdim=True) + param[..., -1].unsqueeze(-1) for enemy_info, param in zip(enemies_h_hiddent_3, w_and_b_attack)]

        q = th.cat([wo_action_fc_Q, *attack_enemy_id_Q], dim=-1)
        hidden_state = th.cat([h_env, *enemies_h_hiddent_3], dim=-1)

        return q, hidden_state
