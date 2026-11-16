
"""@package mlflow_skorch_callbacks.py
This file contains skorch callbacks to log in mlflow.
"""


import mlflow
from skorch.callbacks import Callback
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utl.logging import Log as log


#            mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=None)

class ParamLogger(Callback):
    ''' Log training parameters.
    '''
    def on_train_begin(self, net, X, y):
        # Log all parameters to MLFlow
        for name, param in net.get_params().items():
            mlflow.log_param(name, param)

#class AppMLFlowLogger(MLflowLogger):
#    ''' Log training history and models.
#    '''



class PredictionHistory(Callback):
    """
    Log intermediate results in a pre-allocated dict with format:
    d = { "y_hat" : list(), "latent" : list(), "loss":list(), "epoch":list()}
    """
    def __init__(self, history_dict=None, print_every=100):

        self.history_dict = history_dict 
        self.print_every = print_every

        self.y_hat_history_ = None
        self.latent_history_ = None
        self.loss_history_ = None

        self.first_epoch = True

    def initialize(self):

        log.info(f"INSIDE initialize PREDICTION HISTORY CALLBACK" )

        self.first_epoch = True

        self.history_dict.clear()

        self.history_dict["y_hat"] = list()
        self.history_dict["latent"] = list()
        self.history_dict["loss"] = list()
        self.history_dict["epoch"] = list()

        self.y_hat_history_ = self.history_dict["y_hat"]
        self.latent_history_ = self.history_dict["latent"]
        self.loss_history_ = self.history_dict["loss"]
        self.epoch_history_ = self.history_dict["epoch"]

        return self

    def on_epoch_end(self, net, **kwargs):

        epoch_id = len(net.history)
        if not self.first_epoch and epoch_id % self.print_every != 0:
            return

        self.first_epoch = False

        with torch.no_grad():

            y_hat = net.module_.generator( net.module_.latent)

            self.y_hat_history_.append( y_hat.detach().cpu().clone())

            self.latent_history_.append( net.module_.latent.detach().cpu().clone())

        self.loss_history_.append( net.history[-1]["train_loss"])
        self.epoch_history_.append(epoch_id)



class CustomGradientClipping(Callback):
    def __init__(self, clip_value=None):
        self.clip_value = clip_value

    def on_grad_computed(self, net, named_parameters, **kwargs):
        # Se clip_value for None, o grid search ignora o clipping
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in named_parameters if p.grad is not None], 
                max_norm=self.clip_value
            )


# 2. Esta é a nova classe que herda da classe base e adiciona o monitoramento
class MonitoredGradClip(CustomGradientClipping):
    def __init__(self, clip_value=1.0, print_every=5, iterations=0):
        # Inicializa a classe base herdando o clip_value
        super().__init__(clip_value=clip_value)
        self.print_every = print_every
        self.iterations = iterations  # Contador de lotes (batches)

    def on_grad_computed(self, net, named_parameters, **kwargs):

        # Executa o print apenas no intervalo definido (padrão: a cada 50 batches)
        should_print = (self.iterations % self.print_every == 0)

        if should_print:
            print(f"\n[AUDITORIA CLIP] Batch #{self.iterations} | clip_value is {self.clip_value}")

        if self.clip_value is None:
            return

        # Filtra os parâmetros com gradiente válido
        params = [p for _, p in named_parameters if p.grad is not None]
        if not params:
            return

        if should_print:
            # Calcula a norma L2 total antes da modificação do PyTorch
            norma_antes = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
            ).item()

        # Chama a lógica de clipping da classe base que criamos anteriormente
        super().on_grad_computed(net, named_parameters, **kwargs)

        if should_print:
            # Calcula a norma L2 total após a execução do clipping
            norma_depois = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
            ).item()
            
            print(f"\n[AUDITORIA CLIP] Batch #{self.iterations} | "
                  f"Norma Original: {norma_antes:.4f} | "
                  f"Norma Pós-Clip (Limite: {self.clip_value}): {norma_depois:.4f}")

        # Incrementa o contador de iterações do treinamento
        self.iterations += 1


class PrintLRScheduler(Callback):
    def on_epoch_end(self, net, **kwargs):
        lr = net.optimizer_.param_groups[0]['lr']
        log.info(f"Epoch {len(net.history)} - Learning Rate: {lr:.6f}")


class TwoPhaseTraining(Callback):
    def __init__(self, nmb_of_epochs=500, step1_perc=0.5, warmup_perc=0.25, finetune_lr=1e-5):
        self.step1_epochs = nmb_of_epochs * step1_perc
        self.warmup_epochs = nmb_of_epochs * warmup_perc
        self.finetune_lr = finetune_lr
        self.is_first_time = True

    def on_epoch_begin(self, net, **kwargs):
        # In skorch, len(net.history) gives the current 0-indexed epoch number
        # because the history only appends after an epoch finishes.
        current_epoch = len(net.history)

        # ==========================================
        # TRIGGER STEP 1: Initialization 
        # ==========================================
        if self.is_first_time:
            self.is_first_time = False

            log.info(f"[Step 1] Epoch {current_epoch}: Skips OFF, training pure bottleneck...")
            
            # 1. Force skip connections OFF
            net.module_.use_skip_connections = False
            
            # 2. Ensure all layers are fully trainable
            for param in net.module_.parameters():
                param.requires_grad = True
                
        elif current_epoch == self.step1_epochs:
            log.info(f"\n[Step 2 - Phase A] Epoch {current_epoch}: Skips ON, Freezing Encoder/Latent...")
            
            # 1. Activate skip connections
            net.module_.use_skip_connections = True
            
            # 2. Freeze Encoder and Latent blocks
            for param in net.module_.encoder.parameters():
                param.requires_grad = False
            for param in net.module_.latent.parameters():
                param.requires_grad = False
                
            # Ensure Decoder is trainable
            for param in net.module_.decoder.parameters():
                param.requires_grad = True
                
            # 3. CRITICAL: Re-initialize the optimizer. 
            # Skorch automatically filters for requires_grad=True during this call,
            # and it flushes all historical momentum from AdamW.
            net.initialize_optimizer()

        # ==========================================
        # TRIGGER PHASE B: Full Fine-tuning
        # ==========================================
        elif current_epoch == self.step1_epochs + self.warmup_epochs:
            log.info(f"\n[Step 2 - Phase B] Epoch {current_epoch}: Unfreezing all, lowering LR...")
            
            # 1. Unfreeze all layers
            for param in net.module_.parameters():
                param.requires_grad = True
                
            # 2. Lower the global learning rate
            net.lr = self.finetune_lr
            
            # 3. Re-initialize the optimizer to pick up the un-frozen layers 
            # and apply the new learning rate cleanly.
            net.initialize_optimizer()


#class MLflowValidationFigures(Callback):
#
#    def __init__(
#        self,
#        viewer,
#        y_valid,
#        log_every=20,
#        vmin=-1.7,
#        vmax=1.7,
#    ):
#        self.viewer = viewer
#        self.y_valid = y_valid
#        self.log_every = log_every
#        self.vmin = vmin
#        self.vmax = vmax
#
#
#    def on_epoch_end(self, net, **kwargs):
#
#        epoch = net.history[-1, "epoch"]
#
#        # Log first epoch and then every `log_every` epochs.
#        if epoch != 1 and epoch % self.log_every != 0:
#            return
#
#        # Prediction on validation set
#        y_hat = net.predict(net.X_valid)
#
#
#PAREI AQUI
#        # ---------------------------------------------------------
#        # Metrics
#        # ---------------------------------------------------------
#        mae = ...
#        mse = ...
#        ssim = ...
#        ssim_qmap = ...
#
#        samples = list(range(len(mae)))
#
#        # ---------------------------------------------------------
#        # Generate figures
#        # ---------------------------------------------------------
#        yy_mae_fig = self.viewer.yy_view(
#            y=self.y_valid,
#            y_hat=y_hat,
#            loss=mae,
#            samples=samples,
#            vmin=self.vmin,
#            vmax=self.vmax,
#            colorbar=True,
#            title=f"MAE - Validation - Epoch {epoch}",
#        )
#
#        yy_mse_fig = self.viewer.yy_view(
#            y=self.y_valid,
#            y_hat=y_hat,
#            loss=mse,
#            samples=samples,
#            vmin=self.vmin,
#            vmax=self.vmax,
#            colorbar=True,
#            title=f"MSE - Validation - Epoch {epoch}",
#        )
#
#        yyq_ssim_fig = self.viewer.yyq_view(
#            y=self.y_valid,
#            y_hat=y_hat,
#            qmap=ssim_qmap,
#            loss=ssim,
#            samples=samples,
#            vmin=self.vmin,
#            vmax=self.vmax,
#            colorbar=True,
#            title=f"SSIM - Validation - Epoch {epoch}",
#        )
#
#        # ---------------------------------------------------------
#        # Log to MLflow
#        # ---------------------------------------------------------
#        mlflow.log_figure(
#            yy_mae_fig,
#            f"validation/epoch_{epoch:04d}/mae.png",
#        )
#
#        mlflow.log_figure(
#            yy_mse_fig,
#            f"validation/epoch_{epoch:04d}/mse.png",
#        )
#
#        mlflow.log_figure(
#            yyq_ssim_fig,
#            f"validation/epoch_{epoch:04d}/ssim.png",
#        )
#
#        # Close figures to avoid accumulating matplotlib figures
#
#        plt.close(yy_mae_fig)
#        plt.close(yy_mse_fig)
#        plt.close(yyq_ssim_fig)






    
