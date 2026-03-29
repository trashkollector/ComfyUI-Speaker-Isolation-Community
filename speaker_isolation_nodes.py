import sys
import torch
import torchaudio
import numpy as np
from comfy import model_management as mm


################################################################################################################
########### THIS IS A FORK FROM               https://github.com/pmarmotte2/ComfyUI-Speaker-Isolation
############ NEED TO ACCPEPT TERMS on website https://huggingface.co/pyannote/speaker-diarization-community-1
############ NEED A TOKEN FROM HUGGING FACE
##################################################################################################################
class IterateThruSpeakers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("total_segments", "start_time", "duration")
    FUNCTION = "iterateThruSpeakers"
    CATEGORY = "audio"

    def iterateThruSpeakers(self, audio, hf_token, index):
        print(f"[IterateThruSpeakers] Starting… v1.0 | Requested index: {index}")

        try:
            import sys
            import torch
            import torchaudio

            # Device
            processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Prepare waveform
            sr = audio["sample_rate"]
            wf = audio["waveform"]

            if wf.ndim == 3:
                mono = wf[0].mean(dim=0)
            elif wf.ndim == 2:
                mono = wf[0]
            else:
                mono = wf
            mono = mono.cpu()

            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                mono = resampler(mono)

            audio_for_diar = {"waveform": mono.unsqueeze(0), "sample_rate": target_sr}

            # Diarization
            from pyannote.audio import Pipeline
            ##################################################################
            ############ NEED TO ACCPEPT TERMS on website https://huggingface.co/pyannote/speaker-diarization-community-1
            ############ NEED A TOKEN FROM HUGGING FACE
            ####################################################################
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
            pipeline.to(processing_device)
            diarization = pipeline(audio_for_diar)

            # Collect all segments in chronological order
            raw_segments = []
            for turn, speaker in diarization.speaker_diarization:
                raw_segments.append((float(turn.start), float(turn.end), speaker))

            # Sort by start time
            raw_segments.sort(key=lambda x: x[0])

            # Merge contiguous blocks of the same speaker
            merged = []
            for start, end, speaker in raw_segments:
                if merged and merged[-1][2] == speaker:
                    # Same speaker as last block — extend it
                    merged[-1] = (merged[-1][0], end, speaker)
                else:
                    # Different speaker — new block
                    merged.append((start, end, speaker))

            total_segments = len(merged)
            print(f"[SpeakerSegmentInfoNode] Total merged segments: {total_segments}")
            for i, (s, e, spk) in enumerate(merged, 1):
                print(f"  Block {i}: {spk} | {s:.2f}s → {e:.2f}s | duration: {e-s:.2f}s")

            # Validate index
            if index < 1 or index > total_segments:
                print(f"[SpeakerSegmentInfoNode] Index {index} out of range (1-{total_segments}), returning zeros")
                return (total_segments, 0.0, 0.0)

            # Get requested block
            currSpeakerIdx = index-1
            nextSpeakerIdx = index
            seg_start, seg_end, seg_speaker = merged[currSpeakerIdx]
            duration = seg_end - seg_start 

            # Add silence until next speaker starts
            if nextSpeakerIdx < total_segments:
                next_start, next_end, next_speaker = merged[nextSpeakerIdx]
                new_duration = next_start - seg_start - 0.1
                if new_duration > duration:
                    duration = new_duration



            print(f"[SpeakerSegmentInfoNode] Returning block {index}: {seg_speaker} | start={seg_start:.2f}s | duration={duration:.2f}s")

            return (total_segments, float(seg_start), float(duration))

        except Exception as e:
            import traceback
            print(f"[SpeakerSegmentInfoNode] Error: {str(e)}\n{traceback.format_exc()}")
            return (0, 0.0, 0.0)
        


class SpeakerDiarizerChronoNode:
    """
    Speaker diarization for ComfyUI with guaranteed chronological ordering:
    speaker_1_audio = first person to speak, speaker_2_audio = second, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": "Hugging Face token for pyannote.audio"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "Compute device"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("speaker_1_audio", "speaker_2_audio", "speaker_3_audio", "speaker_4_audio", "summary")
    FUNCTION = "diarize_audio"
    CATEGORY = "Audio/Isolation"

    def _silent_outputs(self, audio, msg):
        sr = audio["sample_rate"]
        wf = audio["waveform"]
        samples = wf.shape[-1]
        silent = {"waveform": torch.zeros((1, 1, samples)), "sample_rate": sr}
        return silent, silent, silent, silent, msg

    def diarize_audio(self, audio, hf_token, device):
        # Make logs easy to identify this new node
        print("[SpeakerDiarizerChronoNode] Starting… v1.1")
        try:
            sys.setrecursionlimit(3000)
            #print("[SpeakerDiarizerChronoNode] Recursion limit set to", sys.getrecursionlimit())
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception as e:
            print(f"[SpeakerDiarizerChronoNode] Warning: cannot restrict threads: {e}")

        # Device selection
        if device == "auto":
            processing_device = mm.get_torch_device()
        elif device == "cuda":
            processing_device = torch.device("cuda")
        else:
            processing_device = torch.device("cpu")

        # Prepare waveform
        sr = audio["sample_rate"]
        wf = audio["waveform"]
        print(f"[SpeakerDiarizerChronoNode] Original waveform {wf.shape} @ {sr}Hz")

        if wf.ndim == 3:      # (B, C, S)
            mono = wf[0].mean(dim=0)
        elif wf.ndim == 2:    # (C, S)
            mono = wf[0]
        else:                 # (S,)
            mono = wf
        mono = mono.cpu()

        target_sr = 16000
        if sr != target_sr:
            print(f"[SpeakerDiarizerChronoNode] Resampling {sr} → {target_sr}")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            mono = resampler(mono)

        audio_for_diar = {"waveform": mono.unsqueeze(0), "sample_rate": target_sr}
        print(f"[SpeakerDiarizerChronoNode] Waveform for diarization: {audio_for_diar['waveform'].shape}")

        # Diarization
        try:
            from pyannote.audio import Pipeline
            print(f"[SpeakerDiarizerChronoNode] Loading pyannote pipeline on {processing_device}")
            ##################################################################################
            ############# MAKE SURE YOU HAVE A TOKEN + ACCEPT TERMS ON WEBSITE or this will not work
            pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=hf_token)

            pipeline.to(processing_device)

            diarization = pipeline(audio_for_diar)
            print("[SpeakerDiarizerChronoNode] Diarization result:\n", diarization)
        except Exception as e:
            import traceback
            err = f"Error during diarization: {str(e)}\n{traceback.format_exc()}"
            print("[SpeakerDiarizerChronoNode]", err)
            return self._silent_outputs(audio, err)


        # Post-processing with strict chronological order
        try:
            # Collect segments per label (DO NOT use alphabetical label order anywhere)
            speaker_segments = {}

            for turn, speaker in diarization.speaker_diarization:
                speaker_segments.setdefault(speaker, []).append((float(turn.start), float(turn.end)))

            # Add this to see ALL segments per speaker
            for lab, segs in speaker_segments.items():
                print(f"[SpeakerDiarizerChronoNode] {lab} segments: {segs}")


            # Compute first start per speaker and sort
            speaker_first_start = {lab: min(s[0] for s in segs) for lab, segs in speaker_segments.items()}
            speakers_ordered = sorted(speaker_first_start.keys(), key=lambda k: speaker_first_start[k])

            print("[SpeakerDiarizerChronoNode] Chronological mapping (this defines output order):")
            for i, lab in enumerate(speakers_ordered, 1):
                print(f"  Output {i} -> {lab} @ {speaker_first_start[lab]:.2f}s")


            # Prepare original sample rate waveform for output building
            wf = audio["waveform"]
            if wf.ndim == 3:
                src = wf[0].mean(dim=0)
            elif wf.ndim == 2:
                src = wf[0]
            else:
                src = wf
            samples = src.shape[0]
            sr = audio["sample_rate"]

            outputs = []
            # Build tracks strictly following speakers_ordered
            for i, lab in enumerate(speakers_ordered[:4]):
                spk_wf = torch.zeros_like(src)
                for start_t, end_t in speaker_segments[lab]:
                    start = int(start_t * sr)
                    end = int(end_t * sr)
                    start = max(0, min(start, samples))
                    end = max(0, min(end, samples))
                    if end > start:
                        spk_wf[start:end] = src[start:end]
                outputs.append({"waveform": spk_wf.unsqueeze(0).unsqueeze(0), "sample_rate": sr})
                print(f"[SpeakerDiarizerChronoNode] Built output {i+1} for {lab}")

            # Pad remaining outputs with silence
            while len(outputs) < 4:
                outputs.append({"waveform": torch.zeros_like(src).unsqueeze(0).unsqueeze(0), "sample_rate": sr})

            summary_lines = []
            for i, lab in enumerate(speakers_ordered):
                segs_str = ", ".join([f"{s:.2f}s-{e:.2f}s" for s, e in speaker_segments[lab]])
                summary_lines.append(f"Output {i+1} -> {lab}: {segs_str}")
            

            summary = "Speakers ordered by first appearance:\n" + "\n".join(summary_lines)

            return tuple(outputs) + (summary,)

        except Exception as e:
            import traceback
            err = f"Error in postprocessing: {str(e)}\n{traceback.format_exc()}"
            print("[SpeakerDiarizerChronoNode]", err)
            return self._silent_outputs(audio, err)

