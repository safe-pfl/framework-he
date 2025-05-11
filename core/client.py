from core.federated_base import FederatedBase
from utils.log import Log
from validators.config_validator import ConfigValidator
import copy as py_copy
from core.train import train
import torch
import torch.nn.functional as F
from torch import autograd

class Client(FederatedBase):
    def __init__(self, model, optimizer_fn, id_num, train_data_loader, evaluation_data_loader, config: 'ConfigValidator', log: 'Log'):
        super().__init__(model, config, log)

        self.optimizer = optimizer_fn(self.model.parameters())

        self.train_loader = train_data_loader
        self.eval_loader = evaluation_data_loader

        self.gradients = {}
        self.id = id_num

        self.log.info(f"client no: {self.id} initialized")

    def synchronize_with_server(self, server):
        self.model.load_state_dict(server.model.state_dict())


    def compute_weight_update(
            self,
            be_ready_for_clustering,
            epochs=None,
            loader=None,
    ):

        epochs = self.config.NUMBER_OF_EPOCHS if epochs is None else epochs
        _updated_model, train_stats = train(
            self.model,
            self.train_loader if not loader else loader,
            self.optimizer,
            epochs,
            self.device,
            self.log,
        )

        self.model.load_state_dict(_updated_model.state_dict())
        del _updated_model

        self.log.info(f"training done for client no {self.id} with loss of {train_stats}")

        if be_ready_for_clustering:
            criterion = torch.nn.CrossEntropyLoss().to(device=self.device, non_blocking=True)

            _model = py_copy.deepcopy(self.model)
            _model.eval()

            accumulated_grads = []
            for param in _model.parameters():
                if param.requires_grad:
                    accumulated_grads.append(torch.zeros_like(param, device=self.device))
                else:
                    accumulated_grads.append(None)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = _model(inputs)
                loss = criterion(outputs, labels.long())

                grads = torch.autograd.grad(loss, _model.parameters(), allow_unused=True)

                for i, grad in enumerate(grads):
                    if grad is not None:
                        accumulated_grads[i] += grad.detach().abs()

            all_grads = []
            for grad in accumulated_grads:
                if grad is not None:
                    all_grads.append(grad.view(-1).cpu())

            if all_grads:
                combined_grads = torch.cat(all_grads).numpy()
                self.gradients = {i: val for i, val in enumerate(combined_grads)}
                self.log.info(f"Gradients computed with {len(self.gradients)} entries.")
            else:
                self.log.warn("No gradients were computed.")
                self.gradients = {}

            del _model
        return train_stats