from typing import Dict
from reILs.infrastructure.utils import pytorch_util as ptu

class GAILPolicy(PPOPolicy):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.disc_net =  \
            ptu.build_mlp(
                input_size=self.obs_dim + self.act_dim,
                output_size=1,
                layers=self.layers
            )
        self.disc_net.apply(ptu.init_weights)
        self.disc_optimizer = optim.Adam()
            self.disc_net.parameters(),
            self.learning_rate,
        )
        self.disc_update_num = config.get('disc_update_num', 1)

    def update(
        self,
        batch: Batch = None,
        **kwargs: Any
    )-> Dict[str, List[float]]:
