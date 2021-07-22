import torch as th

class Hyperparameters:
    def __init__(self):
        self.learning_rate = 0.001

    def create_learning_rate(self, idx):
        return self.learning_rate/(2**idx)

    @staticmethod
    def create_net(net_size, idx):
        net = []
        for i in range(net_size):
            net.append(2**(5+idx))
        return net

    def create_policy_kwargs(self, actor_size, actor_idx, critic_size, critic_idx):
        pi_net = self.create_net(actor_size, actor_idx)
        vf_net = self.create_net(critic_size, critic_idx)
        policy_args = dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=pi_net,
                                          vf=vf_net)])
        policy_info = {
            "pi": pi_net,
            "vf": vf_net
        }

        return policy_args, policy_info
