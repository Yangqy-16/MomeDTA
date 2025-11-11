from .downstream import *


@MODULE_REGISTRY.register()
class FusionModule(DownstreamModule):
    def forward(self, 
        batch: tuple[dict[str, Tensor], Tensor, list[str], list[str]]
    ) -> tuple[Tensor, Tensor, Tensor, list[str], list[str]]:
        data, label, drug, prot = batch

        for name, tensor in data.items():
            if tensor is not None:
                data[name] = tensor.cuda()

        if self.cfg.MODEL.USE_1D and self.cfg.MODEL.USE_2D and self.cfg.MODEL.USE_3D:
            logit, w, att1, att3 = self.model(data) #
        else:
            logit = self.model(data)
        loss = self.criterion(logit, label)

        return loss, logit, label, drug, prot, w, att1, att3
