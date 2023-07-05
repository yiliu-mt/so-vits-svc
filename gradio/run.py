import argparse
import json
import yaml
import torch
import torchaudio
import gradio as gr
import numpy as np
from inference.infer_tool import Svc, RealTimeVC
import soundfile as sf
import io
import tempfile

import logging
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),  # 将日志输出到控制台
            logging.FileHandler('gradio.log'),
        ],
)
# logging = logging.getLogger(__name__)
# from loguru import logger as logging


def load_speaker_ids_from_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    speaker_ids = config.get("spk", {}).keys()
    assert len(speaker_ids) > 0, f'speaker ids not found in {config_path}'
    logging.info(f'load speaker_ids {speaker_ids}')
    return list(speaker_ids)


def prepare_input_wav_in_mem(audio):
    sr, data = audio
    logging.info(f'audio sr={sr} len={len(data)} dur={len(data) / sr:.3f}')
    input_wav_path = io.BytesIO()
    with sf.SoundFile(input_wav_path, 'wb', sr, channels=1, format='wav') as f:
        f.write(data)
    input_wav_path.seek(0)
    return input_wav_path


def warmup_svc(svc_model, kwarg):
    logging.info('Warming up the model...')
    speaker = 0
    tran = kwarg.get('trans', 0)
    auto_predict_f0 = kwarg['auto_predict_f0']
    f0_predictor = kwarg['f0_predictor']
    audio = (svc_model.target_sample, np.zeros(svc_model.target_sample, dtype=np.float32))
    raw_path = prepare_input_wav_in_mem(audio)
    svc_model.infer(
        speaker=speaker,
        tran=tran,
        raw_path=raw_path,
        auto_predict_f0=auto_predict_f0
    )
    return



class GradioInfer:
    def __init__(
        self,
        use_http_frontend,
        port,
        config,
        model,
        cluster_model_path,
        kwarg,
        title,
        description,
        article,
        example_inputs
    ):
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        self.config = config
        self.model = model
        self.port = port
        self.use_http_frontend = use_http_frontend
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'load svc config from {config}')
        logging.info(f'load svc model from {model}')
        self.speaker_ids = load_speaker_ids_from_config(config)
        self.svc_model = Svc(model, config, cluster_model_path=cluster_model_path, device=self.device)
        self.svc = RealTimeVC()
        self.kwarg = kwarg

        warmup_svc(self.svc_model, kwarg)
 
    def greet(self, speaker_id, f_pitch_change, auto_predict_f0, output_sample, audio, raw_infer=True, format='wav'):

        svc_model = self.svc_model
        input_wav_path = prepare_input_wav_in_mem(audio)
        trans = float(f_pitch_change)
        output_sample = int(output_sample)
        kwarg = self.kwarg.copy()
        kwarg['auto_predict_f0'] = auto_predict_f0

        # 模型推理
        if raw_infer:
            if 'slice_db' in kwarg:
                logging.info(f'infer slice')
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                    tmp_file.write(input_wav_path.read())
                    tmp_file.flush()
                    out_audio = svc_model.slice_inference(tmp_file.name, speaker_id, trans, **kwarg)
                    out_audio = torch.from_numpy(out_audio)
            else:
                logging.info(f'infer raw')
                out_audio, _, _ = svc_model.infer(speaker_id, trans, input_wav_path, **kwarg)
                out_audio = out_audio.cpu()
        else:
            logging.info(f'infer realtime')
            out_audio = self.svc.process(speaker_id, trans, input_wav_path, **kwarg)
            out_audio = torch.from_numpy(out_audio)

        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, output_sample).numpy()
        # logging.info(f'tar_audio {tar_audio.dtype}')  # np.float32
        tar_audio = np.int16(tar_audio * np.iinfo(np.int16).max).astype(np.int16)
        return output_sample, tar_audio

    def run(self):
        examples = [
                ['SSB3000', 0, True, '44100', self.example_inputs[0]],
                ['SSB3000', 1.0, True, '16000', self.example_inputs[1]],
                ]
        speaker_ids = self.speaker_ids
        output_sample = self.svc_model.target_sample
        sample_choices = list(map(str, [16000, 24000, 44100, 48000]))
        microphone_input = gr.components.Audio(source='microphone', type='numpy', label='Microphone')
        audio_file_input = gr.components.Audio(source='upload', type='numpy', label='Upload Audio')
        vc_rec = gr.Interface(fn=self.greet,
                             inputs=[
                                gr.components.Dropdown(choices=speaker_ids, value='SSB3000', label="target speaker id"),
                                gr.components.Number(value=0, label="f0 音高调整，支持正负（半音）"),
                                gr.components.Checkbox(value=True, label="auto_predict_f0", info="自动预测 Pitch？(推荐说话启用，唱歌可以不需要)"),
                                gr.components.Dropdown(choices=sample_choices, value=sample_choices[2], label="output_sample"),
                                microphone_input,
                             ],
                             outputs=gr.components.Audio(),
                             allow_flagging="never",
                             title=self.title,
                             description=self.description,
                             article=self.article,
                             # examples=examples,
                             # examples_per_page=5,
                             )
        vc_file = gr.Interface(
                fn=self.greet,
                             inputs=[
                                gr.components.Dropdown(choices=speaker_ids, value='SSB3000', label="target speaker id"),
                                gr.components.Number(value=0, label="f0 音高调整，支持正负（半音）"),
                                gr.components.Checkbox(value=True, label="auto_predict_f0", info="自动预测 Pitch？(推荐说话启用，唱歌可以不需要)"),
                                gr.components.Dropdown(choices=sample_choices, value=sample_choices[2], label="output_sample"),
                                audio_file_input,
                             ],
                             outputs=gr.components.Audio(),
                             allow_flagging="never",
                             title=self.title,
                             description=self.description,
                             article=self.article,
                             examples=examples,
                             examples_per_page=5,
                             )
        iface = gr.TabbedInterface([vc_file, vc_rec], ['VC from File', 'VC from Microphone'])
        iface.launch(share=False, server_port=self.port, server_name="0.0.0.0", enable_queue=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_http_frontend", action="store_true", help="If given, use HTTP frontend")
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--config")
    parser.add_argument("--model")

    parser.add_argument('-cl', '--clip', type=float, default=0, help='音频强制切片，默认0为自动切片，单位为秒/s')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
    
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="dio", help='选择F0预测器,可选择crepe,pm,dio,harvest,默认为pm(注意：crepe为原F0使用均值滤波器)')
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='语音转换自动预测音高，转换歌声时不要打开这个会严重跑调')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='聚类模型或特征检索索引路径，如果没有训练聚类或特征检索则随便填')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='是否使用角色融合')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例')

    # 浅扩散设置
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='扩散模型路径')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='扩散模型配置文件路径')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='扩散步数，越大越接近扩散模型的结果，默认100')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='纯扩散模式，该模式不会加载sovits模型，以扩散模型推理')

    # 不用动的部分
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50')
    parser.add_argument('-d', '--device', type=str, default=None, help='推理设备，None则为自动选择cpu和gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='噪音级别，会影响咬字和音质，较为玄学')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='音频输出格式')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='使增强器适应更高的音域(单位为半音数)|默认为0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音')

    args = parser.parse_args()

    trans = args.trans
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    kwarg = {
        "slice_db" : slice_db,
        "cluster_infer_ratio" : cluster_infer_ratio,
        "auto_predict_f0" : auto_predict_f0,
        "noice_scale" : noice_scale,
        "pad_seconds" : pad_seconds,
        "clip_seconds" : clip,
        "lg_num": lg,
        "lgr_num" : lgr,
        "f0_predictor" : f0p,
        "enhancer_adaptive_key" : enhancer_adaptive_key,
        "cr_threshold" : cr_threshold,
        "k_step":k_step,
        "use_spk_mix":use_spk_mix,
        "second_encoding":second_encoding,
        "loudness_envelope_adjustment":loudness_envelope_adjustment
    }
    logging.info(f'svc kwarg: {kwarg}')

    gradio_config = yaml.safe_load(open('gradio/config/gradio_settings.yaml'))

    g = GradioInfer(
        use_http_frontend=args.use_http_frontend,
        port=args.port,
        config=args.config,
        model=args.model,
        cluster_model_path=args.cluster_model_path,
        kwarg = kwarg,
        **gradio_config)
    g.run()
