import lightning as l
import sentencepiece as spm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.dataset import create_or_load_tokenizer
from src.model import BERT
from src.tasks import MaskedLanguageModel, NextSentencePrediction


class BERTPretrainModel(l.LightningModule):
    def __init__(
        self,
        arg: DictConfig
        ) -> None:
        super().__init__()
        self.arg = arg
        self.vocab = self.get_vocab()
        self.model = self.get_model()
        self.mlm_task = MaskedLanguageModel(
            hidden_dim=self.arg.model.hidden_dim,
            vocab_size=self.vocab.get_piece_size(),
        )
        self.nsp_task = NextSentencePrediction(
            hidden_dim=self.arg.model.hidden_dim,
        )
        self.loss_function = nn.NLLLoss(
            ignore_index=self.vocab.PieceToId(self.arg.model.pad_token)
        )
    
    def _shared_eval_step(self, batch: any, batch_ids: int) -> Tensor:
        src_input, trg_input, trg_output = batch
        output = self.model(src_input, trg_input)
        
        mlm_output = self.mlm_task(output)
        nsp_output = self.nsp_task(output)
        
        return self.calculate_loss(mlm_output, nsp_output, trg_output)
        
    def training_step(self, batch: any, batch_idx: int) -> dict[str, Tensor]:
        src_input, target_sen, seg_token = batch
        output = self.model(src_input, seg_token)
        
        mlm_output = self.mlm_task(output)
        nsp_output = self.nsp_task(output)
        
        loss, mlm_loss, nsp_loss = self.calculate_loss(mlm_output, nsp_output, target_sen)
        
        metrics = {"loss": loss, "mlm_loss": mlm_loss, "nsp_loss": nsp_loss}
        self.log_dict(metrics)      # nn.Module 상속

        return metrics

    def validation_step(self, batch: any, batch_idx: int) -> dict[str, Tensor]:
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"loss": loss}
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self) -> any:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param 'optimizer' must be one of [Adam, AdamW].")

        return optimizer

    def calculate_loss(self, mlm_output: Tensor, nsp_output: Tensor, target_sen: Tensor) -> Tensor:
        if self.device == "mps":
            mlm_output = mlm_output.to(device="cpu")
            nsp_output = nsp_output.to(device="cpu")
            target_sen = target_sen.to(device="cpu")
            
        mlm_loss = self.loss_function(mlm_output.transpose(1, 2), target_sen)      # NOTE: why? 
        nsp_loss = self.loss_function(nsp_output, target_sen[:, 0])
        
        loss = mlm_loss + nsp_loss
        
        return loss, mlm_loss, nsp_loss

    def get_model(self) -> nn.Module:
        params = {
        "vocab_size": self.arg.data.vocab_size,
        "hidden_dim": self.arg.model.hidden_dim,
        "n_heads": self.arg.model.n_heads,
        "ff_dim": self.arg.model.hidden_dim * 4,
        "n_layers": self.arg.model.n_layers,
        "max_seq_len": self.arg.model.max_seq_len,
        "dropout_rate": self.arg.model.dropout_rate,
        "padding_id": self.vocab.PieceToId(self.arg.model.pad_token),
        }
        return BERT(**params)


    def get_vocab(
        self,
        ) -> tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        vocab = create_or_load_tokenizer(
        file_path=self.arg.data.train_path,
        save_path=self.arg.data.dictionary_path,
        language=self.arg.data.language,
        vocab_size=self.arg.data.vocab_size,
        tokenizer_type=self.arg.data.tokenizer_type,
        cls_token=self.arg.model.cls_token,
        sep_token=self.arg.model.sep_token,
        mask_token=self.arg.model.mask_token,
        pad_token=self.arg.model.pad_token
        )
        return vocab