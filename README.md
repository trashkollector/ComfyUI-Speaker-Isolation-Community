# ComfyUI-Speaker-Isolation-Community

A custom node for ComfyUI that performs speaker diarization to isolate individual speaker audio tracks from a single audio source.
Forked to give ability to iterate thru all the speakers in the audio.
This can be used in for loop in comfy, to process each segment.

The version uses the Community version of 
https://huggingface.co/pyannote/speaker-diarization-community-1

Housekeeping: You have to get a Hugging face token, you have to accept all terms and conditions so can use the models.

You must accept all terms, there are multiple terms you need accept for each model.  
Go to the website and accept all the terms they have, if you only accept 1 model it will not work.
You will need a Hugging face token for this to work


## Features

-   Takes a single audio input and finds all the speakers.
-   Uses `pyannote.audio` for speaker diarization.
-
-   FORKED ENHANCEMENTS
-   --------------------
-   New node to Iterate thru the speakers in the audio
-   The purpose is to be able to loop thru an audio file and extract all the speakers at the start and end time
-   These audio clips can be used for whatever purposes you need
-   In my workflow example. I am looping thru the audio then generating video from the audio segment
-   then combining all the video and audio segments together to create one long video.

-   I had to make some updates to the existing diarization code to use the new 4.x lib
-   Thank you Claude for the suggestions to refactor parts of the code.


## Node: IterateThruSpeakers

-   **Category:** `Audio`
-   **Inputs:**
    -   `audio` (AUDIO): The input audio file/data.
    -   `hf_token` (STRING): Your Hugging Face access token. This is **required** to download and use `pyannote.audio` pretrained models. 
    -   `index` (INT): index (1 based, select a speaker)
    You can get a token from [hf.co/settings/tokens](https://hf.co/settings/tokens).
-   **Outputs:**
    -   `total_segments` (int): total number of speaker segments
    -   `start_time` (float): start time of audio segment
    -   `duration` (float): duration of audio segment
    -   `speaker` (string): speaker name such as SPEAKER_00

<img width="1368" height="895" alt="fred" src="https://github.com/user-attachments/assets/587e74f2-3a07-44a0-85ee-7aedc7d0f9eb" />

Workflow coming soon ...  

We took audio from an old sitcom and used this node to chop up the audio into segments.
then we added video to all the audio segments.



https://github.com/user-attachments/assets/37be5839-4bc3-4022-94ee-f2ade767becb


## Node: Speaker Diarizer (Isolation)

-   **Category:** `Audio/Isolation`
-   **Inputs:**
    -   `audio` (AUDIO): The input audio file/data.
    -   `hf_token` (STRING): Your Hugging Face access token. This is **required** to download and use `pyannote.audio` pretrained models. You can get a token from [hf.co/settings/tokens](https://hf.co/settings/tokens).
    -   `device` (COMBO): The device to run the diarization model on (`auto`, `cuda`, `cpu`).
-   **Outputs:**
    -   `speaker_1_audio` (AUDIO): Audio track for the first detected speaker.
    -   `speaker_2_audio` (AUDIO): Audio track for the second detected speaker.
    -   `speaker_3_audio` (AUDIO): Audio track for the third detected speaker.
    -   `speaker_4_audio` (AUDIO): Audio track for the fourth detected speaker.
    -   `diarization_summary` (STRING): A text summary of the diarization process (e.g., number of speakers found, duration per speaker).

If fewer than four speakers are detected, the remaining speaker audio outputs will contain only silence. If an error occurs (e.g., missing token, model download issue), all audio outputs will be silent, and the error will be reported in the `diarization_summary`.

## Installation

1.  **Clone this repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and run:
    ```bash
    git clone https://github.com/trashkollector/ComfyUI-Speaker-Isolation-Community
    ```
    
    Why installing from Comfy Manager may fail.  I have commented out the libs in requirements file.
    You will need to do this part yourself. This is intentional becauase  dependencies could 
    mess up your environment .  This is a possibility with ANY custom  node
    but since the pyannote lib has many dependencies I chose to not include it.
    
    

2.  **Install Dependencies:**
    Navigate into the cloned directory `ComfyUI/custom_nodes/ComfyUI-Speaker-Isolation-Community/` and install the required Python packages:
    ```bash

   
    ***** PLEASE NOTE : I'VE COMMENTED THE libs in requirements.txt ... So it won't automatically install within Comfy

           pip install "pyannote-audio>=4.0.4" 

    ***** There are risks when installing libs which have many dependencies so BE CAREFUL  (consider using --no-deps as a start)
    
    ```
    This will install `pyannote.audio` and its dependencies.
    You also need `ffmpeg` installed on your system, as `pyannote.audio` (and `torchaudio`) may rely on it for loading various audio formats.

3.  **Hugging Face Token and Model Agreement:**
    -   You **must** have a Hugging Face account.
    -   You **must** accept the user conditions for the models used by `pyannote.audio`'s default diarization pipeline. As of writing, these are:
        -   `pyannote/segmentation-3.0` ([link](https://hf.co/pyannote/segmentation-3.0))
        -   `pyannote/speaker-diarization-3.1` ([link](https://hf.co/pyannote/speaker-diarization-community-1))
        -   The underlying speaker embedding model, often `speechbrain/speaker-recognition-ecapa-tdnn` ([link](https://hf.co/speechbrain/speaker-recognition-ecapa-tdnn)) or similar.
        Visit these Hugging Face model pages and accept their terms.
    -   Provide your Hugging Face access token (with read permissions) to the `hf_token` input of the node. This token is primarily used for the *initial download* of the models.
    -   **Offline Usage & Model Caching:** Once the necessary `pyannote.audio` models are downloaded for the first time, they are stored in your local Hugging Face cache (usually located at `~/.cache/huggingface/hub` or a path defined by the `HF_HOME` environment variable). Subsequent runs of this node will use these cached models, allowing for offline operation regarding model access. While the `hf_token` input remains, it may not be actively used for network requests if all models are cached.
    -   **Advanced Offline Setup:** For users who need to manage models in a specific local directory completely separate from the standard Hugging Face cache (e.g., for air-gapped environments after an initial setup elsewhere), `pyannote.audio` does support loading pipelines from local paths. This involves manually downloading all required model and configuration files and structuring them correctly. For more details on this advanced scenario, please refer to the [pyannote.audio FAQ](https://github.com/pyannote/pyannote-audio/blob/develop/FAQ.md#can-i-use-gated-models-and-pipelines-offline). The current version of this node uses the Hugging Face model identifier string, relying on the cache mechanism.

4.  **Restart ComfyUI.**

## Usage

1.  Add the "Speaker Diarizer (Isolation)" node from the "Audio/Isolation" category.
2.  Connect an audio source to the `audio` input.
3.  Enter your Hugging Face token into the `hf_token` field.
4.  Connect the desired `speaker_X_audio` outputs to other audio nodes (e.g., Save Audio, audio input for other processes).
5.  The `diarization_summary` can be connected to a text display node to see the results.

## Troubleshooting
-   **`RecursionError: maximum recursion depth exceeded`**: This can sometimes occur with `pyannote.audio` or its dependencies (like `speechbrain`). The node attempts to mitigate this by increasing Python's recursion limit. If it persists, it might indicate a deeper environment or version conflict.
-   **Errors related to model downloads or `ffmpeg`**: Ensure you have a working internet connection, a valid Hugging Face token, have accepted model user agreements on Hugging Face, and that `ffmpeg` is correctly installed and accessible in your system's PATH.
-   **No speakers detected / Incorrect diarization**: The quality of diarization depends on the audio quality and the `pyannote.audio` model's capabilities. Background noise, very short utterances, or heavy overlap can affect performance.
