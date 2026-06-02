import torch
from torch import Tensor


def compute_grad_norm(parameters, norm_type: float = 2.0) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return Tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class AMPGradScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True, device="cuda"):
        self.enabled = bool(enabled)
        self.device = device
        self._scaler = None
        if self.enabled:
            self._scaler = torch.amp.GradScaler(device=device)

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            parameters=None,
            create_graph=False,
            update_grad=True,
    ):
        # 反向传播:
        # self.enabled=True：说明启用 AMP，先把 loss 放大，再 backward
        # self.enabled=False：普通 backward
        if self.enabled:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else:
            loss.backward(create_graph=create_graph)

        # 是否更新参数:
        # 在梯度累积时，不是每个 batch 都 optimizer.step()。比如：
        # grad_accum_steps = 4时，前 3 个 batch 只 backward，不更新参数；第 4 个 batch 才更新。
        # 所以：
        #   update_grad=False：只 backward，累积梯度
        #   update_grad=True：backward 后执行 optimizer step
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # 如果设置了梯度裁剪，就先把 AMP 放大的梯度还原回来：
                # 这个函数会限制梯度范数不超过 clip_grad，同时返回裁剪前的梯度范数。
                if self.enabled:
                    self._scaler.unscale_(
                        optimizer
                    )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if self.enabled:
                    self._scaler.unscale_(optimizer)
                norm = compute_grad_norm(parameters)
            if self.enabled:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        if not self.enabled:
            return {}
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        if self.enabled and state_dict:
            self._scaler.load_state_dict(state_dict)
