from vencoder.encoder import SpeechEncoder
import torch

from vencoder.conformer.model.conformer_models import ConformerEncoder
from vencoder.conformer.task.EmbedExtractor import ASRModel

from vencoder.conformer.utils.cmvn import load_cmvn, GlobalCMVN
from vencoder.conformer.utils.checkpoint import load_checkpoint

from torch.utils.data import DataLoader
from vencoder.conformer.dataset.dataset import Dataset


data_config = {
    'fbank_conf': {
        'frame_length': 25,
        'frame_shift': 10,
        'num_mel_bins': 80,
    },
    'batch_conf': {
        'batch_size': 5,
        'batch_type': 'static',
    },
    'cmvn_file': 'pretrain/global_cmvn',
    'shuffle': False,
}

model_config = {
    'activation_type': 'swish',
    'attention_dropout_rate': 0.0,
    'attention_heads': 8,
    'cnn_module_kernel': 15,
    'causal': True,
    'cnn_module_norm': 'layer_norm',
    'input_layer': 'conv2d',
    'dropout_rate': 0.0,
    'linear_units': 2048,
    'normalize_before': True,
    'num_blocks': 10,
    'output_size': 512,
    'pos_enc_layer_type': 'rel_pos',
    'positional_dropout_rate': 0.0,
    'selfattention_layer_type': 'rel_selfattn',
    'use_cnn_module': True,
}

embedding_config = {
    'resume_asr_encoder_model': '/nfs2/junhao.xu/asr/models/basemodel_lr0017_acc16_fixdata_addtrycatch_large_new/36avg5.pt',
    # 'use_layer_idx': 7
    'use_layer_idx': 9
}


class ConformerPPGLarge(SpeechEncoder):
    def __init__(self, vec_path="pretrain/conformer_embedding_large_model.pt", device=None):
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        # Step1, build model
        mean, istd = load_cmvn(data_config['cmvn_file'], True)
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
        encoder = ConformerEncoder(data_config['fbank_conf']['num_mel_bins'],
                       global_cmvn=global_cmvn,
                       **model_config)
        embed_extractor = ASRModel(encoder=encoder)
        # Step2, load our Conformer Encoder
        load_checkpoint(embed_extractor, vec_path)  # or embedding_config['resume_asr_encoder_model'])
        self.model = embed_extractor.to(self.dev)

    def encoder(self, wav):
        dataloader = self._load_data(wav)
        with torch.no_grad():
            # ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            # ppg = torch.FloatTensor(ppg[:ppgln,]).to(self.dev)
            # return ppg[None,:,:].transpose(1, 2)

            ppg = self._infer(dataloader)
            # ppg = torch.FloatTensor(ppg).to(self.dev)
        return ppg[None,:,:].transpose(1, 2)

    def _load_data(self, wav):
        # Step0, dataset
        dataset = Dataset(
            data_type='raw',
            data_list_file=[dict(key='abc', wav=wav.unsqueeze(0), sample_rate=16000)],
            conf=data_config,
            partition=False,
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=None
        )
        return data_loader
 
    def _infer(self, data_loader):
        all_embeds = {}
        for batch_idx, batch in enumerate(data_loader):
            keys, feats, feats_lengths = batch
            feats = feats.to(self.dev)
            feats_lengths = feats_lengths.to(self.dev)
            embed_results_hooks, pad_masks = self.model(feats, feats_lengths)
            # to select each key's embedding
            embed_results = embed_results_hooks[embedding_config['use_layer_idx']]
            for key, emb, mask in zip(keys, embed_results, pad_masks):
                mask = mask.to(torch.float32)
                emb = emb[:int(torch.sum(mask)), :]
                all_embeds[key] = {
                    'embedding': emb,
                    'pad_mask': mask
                }
        return all_embeds['abc']['embedding']
 
