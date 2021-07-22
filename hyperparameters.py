import torch as th

class Hyperparameters:
    def create_learning_rate(self, idx):
        learning_rate = 0.001
        return learning_rate/(2**idx)

    def create_net(self, net_size, idx):
        net = []
        for i in range(net_size):
            net.append(2**(5+idx))
        return net

    def create_policy_args(self, actor_size, actor_idx, critic_size, critic_idx):
        return dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=self.create_net(actor_size, actor_idx),
                                   vf=self.create_net(critic_size, critic_idx))])
