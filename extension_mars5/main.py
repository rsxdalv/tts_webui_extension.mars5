import gradio as gr
import librosa
import torch

from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.decorator_apply_torch_seed import decorator_apply_torch_seed
from tts_webui.decorators.decorator_log_generation import decorator_log_generation
from tts_webui.decorators.decorator_save_metadata import decorator_save_metadata
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.decorators.log_function_time import log_function_time
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui


def extension__tts_generation_webui():
    ui()
    return {
        "package_name": "extension_mars5",
        "name": "MARS5",
        "requirements": "git+https://github.com/rsxdalv/extension_mars5@main",
        "description": "MARS5: A novel speech model for insane prosody",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "CAMB.AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/camb-ai/mars5-tts",
        "extension_website": "https://github.com/rsxdalv/extension_mars5",
        "extension_platform_version": "0.0.1",
    }


@manage_model_state("mars5")
def get_mars5(model_name="CAMB-AI/MARS5-TTS"):
    from mars5.inference import Mars5TTS

    return Mars5TTS.from_pretrained(model_name)


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("mars5")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts(text, ref_audio, ref_transcript, **cfg_kwargs):
    from mars5.inference import InferenceConfig as config_class

    mars5 = get_mars5(model_name="CAMB-AI/MARS5-TTS")

    wav, _sr = librosa.load(ref_audio, sr=mars5.sr, mono=True)
    wav = torch.from_numpy(wav)
    if ref_transcript is None or ref_transcript == "":
        cfg_kwargs["deep_clone"] = False
    cfg = config_class(
        **{k: v for k, v in cfg_kwargs.items() if hasattr(config_class, k)}
    )
    _ar_codes, wav_out = mars5.tts(text, wav, ref_transcript, cfg=cfg)

    return {
        "audio_out": (mars5.sr, wav_out.cpu().numpy()),
        # "tokens": _ar_codes,
    }


def ui():
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=3, label="Text to generate")
            generate_btn = gr.Button("Generate")

            temperature = gr.Slider(
                label="Temperature", value=0.7, step=0.05, minimum=0.0, maximum=1.0
            )
            top_k = gr.Slider(label="Top-k", value=200, step=1, minimum=0)
            top_p = gr.Slider(
                label="Top-p", value=0.2, step=0.05, minimum=0.0, maximum=1.0
            )
            typical_p = gr.Slider(
                label="Typical-p", value=1.0, step=0.05, minimum=0.0, maximum=1.0
            )

        with gr.Column():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                ref_transcript = gr.Textbox(label="Reference Transcript")
                deep_clone = gr.Checkbox(label="Deep Clone", value=True)

            with gr.Accordion("Advanced", open=False):
                freq_penalty = gr.Slider(label="Frequency Penalty", value=3)
                presence_penalty = gr.Slider(
                    label="Presence Penalty",
                    value=0.4,
                    step=0.05,
                    minimum=0.0,
                    maximum=1.0,
                )
                rep_penalty_window = gr.Slider(label="Rep Penalty Window", value=80)
                eos_penalty_decay = gr.Slider(
                    label="EOS Penalty Decay",
                    value=0.5,
                    step=0.05,
                    minimum=0.0,
                    maximum=1.0,
                )
                eos_penalty_factor = gr.Slider(label="EOS Penalty Factor", value=1)
                eos_estimated_gen_length_factor = gr.Slider(
                    label="EOS Estimated Gen Length Factor", value=1.0
                )
                timesteps = gr.Slider(label="Timesteps", value=200)
                x_0_temp = gr.Slider(
                    label="X0 Temperature",
                    value=0.7,
                    step=0.05,
                    minimum=0.0,
                    maximum=1.0,
                )
                q0_override_steps = gr.Slider(label="Q0 Override Steps", value=20)
                nar_guidance_w = gr.Slider(label="NAR Guidance W", value=3)
                max_prompt_dur = gr.Slider(label="Max Prompt Duration", value=12)
                generate_max_len_override = gr.Slider(
                    label="Generate Max Len Override", value=-1
                )
                trim_db = gr.Slider(label="Trim DB", value=27)
                beam_width = gr.Slider(label="Beam Width", value=1)
                ref_audio_pad = gr.Slider(label="Ref Audio Pad", value=0)

            with gr.Column():
                use_kv_cache = gr.Checkbox(label="Use KV Cache", value=True)
                unload_model_button("mars5")
                seed, randomize_seed_callback = randomize_seed_ui()

    with gr.Column():
        audio_out = gr.Audio(label="Generated Audio")

    generate_btn.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            fn=tts,
            inputs={
                text: "text",
                ref_audio: "ref_audio",
                ref_transcript: "ref_transcript",
                seed: "seed",
                # InferenceConfig
                temperature: "temperature",
                top_k: "top_k",
                top_p: "top_p",
                typical_p: "typical_p",
                freq_penalty: "freq_penalty",
                presence_penalty: "presence_penalty",
                rep_penalty_window: "rep_penalty_window",
                eos_penalty_decay: "eos_penalty_decay",
                eos_penalty_factor: "eos_penalty_factor",
                eos_estimated_gen_length_factor: "eos_estimated_gen_length_factor",
                timesteps: "timesteps",
                x_0_temp: "x_0_temp",
                q0_override_steps: "q0_override_steps",
                nar_guidance_w: "nar_guidance_w",
                max_prompt_dur: "max_prompt_dur",
                generate_max_len_override: "generate_max_len_override",
                deep_clone: "deep_clone",
                use_kv_cache: "use_kv_cache",
                trim_db: "trim_db",
                beam_width: "beam_width",
                ref_audio_pad: "ref_audio_pad",
            },
            outputs={
                "audio_out": audio_out,
                # "tokens": gr.JSON(visible=False),
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
        ),
        api_name="mars5",
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
