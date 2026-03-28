from .speaker_isolation_nodes import SpeakerDiarizerChronoNode, IterateThruSpeakers

NODE_CLASS_MAPPINGS = {
    "SpeakerDiarizerChronoNode": SpeakerDiarizerChronoNode,
    "IterateThruSpeakers": IterateThruSpeakers,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeakerDiarizerChronoNode": "Speaker Diarizer (Isolation)",
    "IterateThruSpeakers": "IterateThruSpeakers",
}

WEB_DIRECTORY = "./js"  # Only if you actually have a /js folder with UI assets

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

print("ComfyUI-Speaker-Isolation: Loaded SpeakerDiarizerChronoNode")
