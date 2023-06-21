import torch

class ASRModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
                
    def forward(self, speech, speech_lengths):
        assert (speech.shape[0] == speech_lengths.shape[0])
        encoder_out_hooks, encoder_mask = self.encoder(speech, speech_lengths)
        return encoder_out_hooks, encoder_mask
                
