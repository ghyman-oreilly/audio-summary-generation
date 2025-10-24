from elevenlabs import ElevenLabs
from google import genai
from google.genai import types
import keyring
from pathlib import Path
import random
import time
import typer
from typing import List, Literal, Optional, Union
import wave

from prompts import TEXT_SUMMARY_PROMPT, TRANSCRIPT_SYS_INSTRUCTIONS


app = typer.Typer()

# https://ai.google.dev/gemini-api/docs/speech-generation
VOICES_GOOGLE = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat"
]

# https://elevenlabs.io/app/default-voices
# I haven't been able to find a way to use
# community voices. Possibly they need to be
# added to our account, but I don't find evidence
# in the docs that it can be done programmatically.
DEFAULT_VOICE_ONE_ELEVENLABS = 'FGY2WhTYpPnrIDTdsKH5' # Laura
DEFAULT_VOICE_TWO_ELEVENLABS = 'TX3LPaxmHKxFdv7VOQHJ' # Liam

DEFAULT_SPEAKER_ONE_LABEL = 'Speaker 1'
DEFAULT_SPEAKER_TWO_LABEL = 'Speaker 2'

# keychain deets
SERVICE_NAME = "audio_summary_generator"
GEMINI_KEY_USER_NAME = "google_api_key"
ELEVENLABS_KEY_USER_NAME = "elevenlabs_api_key"

@app.command(
    help="""
    Add a shared/community voice, by ID, to our
    library of voices (ElevenLabs).

    Only voices in our user library, which includes default
    voices, can be used.
    """)
def add_elevenlabs_voice(
    voice_id: str,
    custom_name: str
):
    ELEVENLABS_API_KEY = check_api_key(SERVICE_NAME, ELEVENLABS_KEY_USER_NAME)
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    current_page = 0
    continue_paging = True
    voice_found = False

    # validate CLI params
    if not voice_id or not custom_name:
        typer.echo(f"Voice ID and Custom Name params must have valid values. Exiting.")
        raise typer.Exit(code=1)

    # TODO: break out these chunks of logic/flow into separate functions?

    # check if voice already exists in user library
    try:
        client.voices.get(voice_id=voice_id)
        voice_found = True
    except Exception as e:
        if 'voice_not_found' in str(e):
            # expected result
            pass
        else:
            # Handle all other unexpected errors
            raise e

    if voice_found:
        typer.echo(f"Shared voice already exists in user library. Exiting.")
        raise typer.Exit(code=1) 

    voice_found = False
    owner_id = None

    # check for shared voice in community library
    typer.echo(f"Searching for voice in community library...")
    while continue_paging:
        some_shared_voices = client.voices.get_shared(page_size=100, page=current_page)
        if some_shared_voices.voices:
            for voice in some_shared_voices.voices:
                if voice.voice_id == voice_id:
                    voice_found = True
                    owner_id = voice.public_owner_id
                    break
            if voice_found:
                break
            if not some_shared_voices.has_more:
                continue_paging = False
        current_page += 1
        time.sleep(0.5)

    if not voice_found:
        typer.echo(f"Shared voice matching ID {voice_id} not found. Exiting.")
        raise typer.Exit(code=1)

    # add voice to user library
    client.voices.share(
        public_user_id=owner_id,
        voice_id=voice_id,
        new_name=custom_name
    )

    typer.echo(f"Shared voice matching ID {voice_id} added to user library.")
    typer.echo(f"Script complete.")

@app.command(
    help="""
    Generate podcast-style audio summary using the default
    TTS service provider (ElevenLabs). 
    """
)
def generate(
        path_to_pdf: Optional[Path] = typer.Argument(
            None, help=(
                "Provide path to a PDF file to run the full audio-summary generation workflow. "
                "PDF must be less than 20MB."
            )
        ),
        output_dir: Optional[Path] = typer.Option(
            None, help=(
                "Provide directory where output files (text, audio) should be saved. "
                "Defaults to current working directory."
            )
        ),
        text_summary_file: Optional[Path] = typer.Option(
            None, help=(
                "Provide path to text file containing the summary of your PDF. "
                "If providing this text file, the PDF path argument doesn't need to be provided."
            )
        ),
        transcript_file: Optional[Path] = typer.Option(
            None, help=(
                "Provide path to text file containing the transcript for your audio. "
                "If providing this text file, the PDF path argument and text_summary_file options "
                "don't need to be provided."
                "Important: make sure your speakers are prefixed with "
                "'Speaker 1:' and 'Speaker 2:' prefixes, unless you're setting "
                "custom prefixes with the `--speaker-one-prefix` and `--speaker-two-prefix` flags."
            )
        ),
        speaker_one_voice: Optional[str] = typer.Option(
            DEFAULT_VOICE_ONE_ELEVENLABS,
            help=(
                f"Enter an ElevenLabs voice ID. "
                f"Default: {DEFAULT_VOICE_ONE_ELEVENLABS}."
            )
        ),
        speaker_two_voice: Optional[str] = typer.Option(
            DEFAULT_VOICE_TWO_ELEVENLABS,
            help=(
                f"Enter an ElevenLabs voice ID. "
                f"Be sure to select a different voice from you chose for speaker one! "
                f"Default: {DEFAULT_VOICE_TWO_ELEVENLABS}."
            )
        ),
):
    generate_audio_summary(
        path_to_pdf,
        output_dir,
        'elevenlabs',
        text_summary_file,
        transcript_file,
        speaker_one_voice,
        speaker_two_voice,
    )

@app.command(
    help="""
    Generate podcast-style audio summary using an alternative
    TTS service provider (Google). 
    """
)
def generate_w_google(
        path_to_pdf: Optional[Path] = typer.Argument(
            None, help=(
                "Provide path to a PDF file to run the full audio-summary generation workflow. "
                "PDF must be less than 20MB."
            )
        ),
        output_dir: Optional[Path] = typer.Option(
            None, help=(
                "Provide directory where output files (text, audio) should be saved. "
                "Defaults to current working directory."
            )
        ),
        text_summary_file: Optional[Path] = typer.Option(
            None, help=(
                "Provide path to text file containing the summary of your PDF. "
                "If providing this text file, the PDF path argument doesn't need to be provided."
            )
        ),
        transcript_file: Optional[Path] = typer.Option(
            None, help=(
                "Provide path to text file containing the transcript for your audio. "
                "If providing this text file, the PDF path argument and text_summary_file options "
                "don't need to be provided."
                "Important: make sure your speakers are prefixed with "
                "'Speaker 1:' and 'Speaker 2:' prefixes, unless you're setting "
                "custom prefixes with the `--speaker-one-prefix` and `--speaker-two-prefix` flags."
            )
        ),
        speaker_one_voice: Optional[str] = typer.Option(
            None,
            help=(
                f"Choose a voice for speaker one. "
                f"Available options are: {', '.join(VOICES_GOOGLE)} "
                f"Default: random."
            )
        ),
        speaker_two_voice: Optional[str] = typer.Option(
            None,
            help=(
                f"Choose a voice for speaker one. "
                f"Available options are: {', '.join(VOICES_GOOGLE)} "
                f"Be sure to select a different voice from you chose for speaker one! "
                f"Default: random."
            )
        ),
        speaker_one_prefix: Optional[str] = typer.Option(
            None,
            help=(
                f"Choose a prefix/lavel for speaker one. "
                f"Default: {DEFAULT_SPEAKER_ONE_LABEL}. "
                f"Note: If running the script withe the `--transcript_file` flag, "
                "this should match whatever you have in your transcript."
            )
        ),
        speaker_two_prefix: Optional[str] = typer.Option(
            None,
            help=(
                f"Choose a prefix/lavel for speaker two. "
                f"Default: {DEFAULT_SPEAKER_TWO_LABEL}. "
                f"Note: If running the script withe the `--transcript_file` flag, "
                "this should match whatever you have in your transcript."
            )
        ),
):
    generate_audio_summary(
        path_to_pdf,
        output_dir,
        'google',
        text_summary_file,
        transcript_file,
        speaker_one_voice,
        speaker_two_voice,
        speaker_one_prefix,
        speaker_two_prefix
    )

def generate_audio_summary(
    path_to_pdf: Optional[Path],
    output_dir: Optional[Path],
    tts_provider: Literal['google', 'elevenlabs'],
    text_summary_file: Optional[Path] = None,
    transcript_file: Optional[Path] = None,
    speaker_one_voice: Optional[str] = None,
    speaker_two_voice: Optional[str] = None,
    speaker_one_prefix: Optional[str] = None,
    speaker_two_prefix: Optional[str] = None
):
    """
    Generate podcast-style audio summary
    from a text PDF.
    """
    GEMINI_API_KEY = check_api_key(SERVICE_NAME, GEMINI_KEY_USER_NAME)
    ELEVENLABS_API_KEY = None

    if tts_provider == 'elevenlabs':
        ELEVENLABS_API_KEY = check_api_key(SERVICE_NAME, ELEVENLABS_KEY_USER_NAME)
        tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        validate_voices_elevenlabs(tts_client, speaker_one_voice, speaker_two_voice)
    else:
        tts_client = genai.Client(api_key=GEMINI_API_KEY)
        speaker_one_voice, speaker_two_voice = select_voices_google(speaker_one_voice, speaker_two_voice)

        speaker_one_prefix, speaker_two_prefix = clean_and_validate_speaker_labels(
            speaker_one_prefix, speaker_two_prefix
        )

    TEXT_MODEL = 'gemini-2.5-flash'
    TTS_MODEL_GOOGLE = 'gemini-2.5-flash-preview-tts'

    TIMESTAMP = int(time.time())
    
    # check/config output directory
    if output_dir:
        if not dir_is_valid(output_dir):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
    else:
        output_dir = Path.cwd()

    # generate text summary
    if path_to_pdf and not text_summary_file and not transcript_file:
        text_summary = execute_pdf_workflow(path_to_pdf, output_dir, GEMINI_API_KEY, TIMESTAMP, TEXT_MODEL)

    # handle existing/inputted text summary
    if text_summary_file:
        if not file_is_valid(text_summary_file, '.txt'):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
        text_summary = read_text_from_file(text_summary_file)
    
    # generate transcript
    if not transcript_file:
        transcript = execute_transcript_generation_workflow(
            text_summary,
            output_dir,
            GEMINI_API_KEY,
            TRANSCRIPT_SYS_INSTRUCTIONS,
            TIMESTAMP,
            speaker_one_prefix,
            speaker_two_prefix,
            TEXT_MODEL
        )

    # handle existing/inputted transcript
    if transcript_file:
        if not file_is_valid(transcript_file, '.txt'):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
        transcript = read_text_from_file(transcript_file)

    # chunk transcript
    typer.echo("Chunking transcript. This may take a few minutes...")
    if tts_provider == 'elevenlabs':
        transcript_chunks = generate_text_to_dialogue_payloads(transcript, speaker_one_voice, speaker_two_voice)
    else:
        transcript_chunks = chunk_string(transcript, GEMINI_API_KEY, TTS_MODEL_GOOGLE)
    typer.echo(f"Transcript split into {len(transcript_chunks)} chunks.")

    # generate audio from chunks
    typer.echo(
                "Generating audio from transcript chunks. "
                "This could take a while (up to 10 minutes per chunk)..."
            )
    audio_chunk_filepaths: List[Path] = generate_audio_chunks(
        transcript_chunks, 
        TIMESTAMP, 
        output_dir,
        tts_provider=tts_provider,
        tts_client=tts_client,
        speaker_one_voice=speaker_one_voice,
        speaker_two_voice=speaker_two_voice,
        chosen_speaker_one_prefix=speaker_one_prefix,
        chosen_speaker_two_prefix=speaker_two_prefix  
    )

    # combine chunk audio files
    typer.echo("Combining audio chunk files...")
    combined_audio_filepath = Path(output_dir / f"combined_audio_{TIMESTAMP}.wav")
    combine_wav_files(audio_chunk_filepaths, combined_audio_filepath)
    typer.echo(f"Combined audio saved to {str(combined_audio_filepath)}")

    delete_audio_chunks = typer.confirm(
        "Do you wish to delete the partial audio chunks?\n"
        "Choose No if you wish to save them in case they need to be respliced later."
        )

    if delete_audio_chunks:
        delete_files(audio_chunk_filepaths)

    typer.echo("Scripted completed.")


def dir_is_valid(
    path_to_dir: Path
):
    """
    Verify that input dir path is valid
    """
    if not path_to_dir.is_dir():
        typer.echo("Input must be a valid directory.")
        return False
    return True


def file_is_valid(
    path_to_file: Path,
    expected_suffix: Optional[str] = None,
    max_size_mb: Optional[int] = None	
) -> bool:
    """
    Verify that input file is valid
    """
    max_size_bytes = None
    
    if max_size_mb:
        max_size_bytes = max_size_mb * 1024 * 1024
    
    if not path_to_file.is_file():
        typer.echo(f"Input must be a valid file.")
        return False
    elif (
        expected_suffix
        and not path_to_file.suffix.lower() == expected_suffix.lower()
    ):
        typer.echo(f"Input file must be a valid {expected_suffix} file.")
        return False
    elif (
        max_size_mb
        and max_size_bytes
        and path_to_file.stat().st_size >= max_size_bytes
    ):
        typer.echo(f"Input file must be less than {max_size_mb} MB.")
        return False
    
    return True


def infer_with_pdf_document_understanding(
        path_to_pdf: Path,
        prompt: str,
        api_key: str,
        model_name: str = 'gemini-2.5-flash'
) -> str:
    """
    Given a PDF file as corpus, generate a response to a user prompt
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=[
                types.Part.from_bytes(
                    data=path_to_pdf.read_bytes(),
                    mime_type='application/pdf',
                ),
                prompt
            ]
    )
    
    return response.text


def generate_text(
        user_prompt: str,
        api_key: str,
        sys_instrux: Optional[str] = None,
        model_name: str = 'gemini-2.5-flash'
) -> str:
    """
    Generate text, given a user prompt
    and optional system instructions

    user_prompt can be a natural language prompt
    or a text that the system instructions indicate
    the model should transform
    """
    client = genai.Client(api_key=api_key)

    if sys_instrux:
        response = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=sys_instrux
            ),
            contents=user_prompt
        )
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt
        )
    
    return response.text


def write_text_to_file(
    text: str,
    output_path: Path	
):
    """
    Write text string to file.
    """
    with open(output_path, 'w') as f:
        f.write(text)


def read_text_from_file(
    text_filepath: Path
) -> str:
    """
    Read text from text file to string
    """
    with open(text_filepath, 'r') as f:
        text = f.read()
    return text


def generate_text_to_dialogue_payloads(
    transcript: str,
    voice_one_id: str,
    voice_two_id: str,
    char_limit: int = 3000
):
    """
    Split a transcript string into payloads
    for the ElevenLabs Text to Dialog API, remaining
    under a given character limit.

    Character limit for ElevenLabs v3 model
    (required for use with Text to Dialog API) 
    is 3000 characters.
    """
    payloads = []
    snippets = [line for line in transcript.splitlines() if line]
    text_char_count = 0
    payload = []
    for ix, snippet in enumerate(snippets):
        text_char_count = text_char_count + len(snippet)

        if text_char_count >= char_limit:
            text_char_count = 0
            if payload:
                payloads.append(payload)
                payload = []

        if ix == 0 or ix % 2 == 0:
            voice_id_to_use = voice_one_id
        else:
            voice_id_to_use = voice_two_id

        input = { 'text': snippet, 'voice_id': voice_id_to_use }

        payload.append(input)
    
    if payload:
        payloads.append(payload)
    
    return payloads
        


def chunk_string(
    text_string: str,
    api_key: str,
    model_name: str = 'gemini-2.5-flash-preview-tts',
    token_limit: Optional[int] = None
):
    """
    Generate a list of strings from a single string,
    keeping within a specified token limit.

    For use with Google TTS.
    """
    client = genai.Client(api_key=api_key)
    
    if not token_limit:
        token_limit = 3000 # could use a map to allow for various models 
                           # (note that count_tokens API method is NOT reliable)

    chunks = []
    current_chunk = ""
    current_token_count = 0

    lines = text_string.split('\n')

    for line in lines:
        response = client.models.count_tokens(
            model=model_name,
            contents=line
        )
        
        line_token_count = response.total_tokens

        if current_token_count + line_token_count <= token_limit:
            current_chunk += line + '\n'
            current_token_count += line_token_count
        else:
            # Start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
            current_token_count = line_token_count

    # Add the last chunk to the list
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_audio_chunks(
    chunks: Union[List[str], List[List[dict]]],
    timestamp: int,
    output_dir: Path,
    tts_provider: Literal['google', 'elevenlabs'],
    tts_client: Union[genai.Client, ElevenLabs],
    speaker_one_voice: str,
    speaker_two_voice: str,
    chosen_speaker_one_prefix: Optional[str] = None,
    chosen_speaker_two_prefix: Optional[str] = None
):
    """
    Given a list of text chunks or Text to Dialog payloads, generate
    and save audio chunks to file, returning
    list of audio chunk filepaths.
    """
    audio_chunk_filepaths = []

    for i, chunk in enumerate(chunks):
        typer.echo(f"Generating audio chunk {i+1} of {len(chunks)}...")
        audio_chunk_filepath = Path(output_dir / f"audio_chunk_{i:03d}_{timestamp}.wav")
        if tts_provider == 'elevenlabs':
            generate_audio_chunk_from_chunk_elevenlabs(
                chunk, 
                audio_chunk_filepath,
                tts_client=tts_client
            )
        else:
            generate_audio_chunk_from_text_chunk_google(
                chunk, 
                audio_chunk_filepath, 
                tts_client=tts_client, 
                speaker_one_voice=speaker_one_voice,
                speaker_two_voice=speaker_two_voice,
                chosen_speaker_one_prefix=chosen_speaker_one_prefix,
                chosen_speaker_two_prefix=chosen_speaker_two_prefix
            )
        audio_chunk_filepaths.append(audio_chunk_filepath)
        typer.echo(f"Audio chunk saved to {str(audio_chunk_filepath)}...")
    
    return audio_chunk_filepaths


def generate_audio_chunk_from_chunk_elevenlabs(
    payload: list[dict],
    output_file: Path,
    tts_client: ElevenLabs,
    model_id: str = 'eleven_v3'
):
    """
    Given a payload, generate audio using ElevenLabs
    Text to Dialog API
    """
    audio_generator = tts_client.text_to_dialogue.convert(
        model_id=model_id,
        inputs=payload,
        output_format='pcm_24000' # important to use this encoding
                                  # for compatibility with write_audio_data_to_wav_file
    )
    audio_bytes = b"".join(list(audio_generator))
    write_audio_data_to_wav_file(output_file, audio_bytes)

def generate_audio_chunk_from_text_chunk_google(
    text: str,
    output_file: Path,
    tts_client: genai.Client, # Google TTS client
    model_name: str = 'gemini-2.5-flash-preview-tts',
    speaker_one_voice: str = 'Puck',
    speaker_two_voice: str = 'Zephyr',
    chosen_speaker_one_prefix: str = "Speaker 1",
    chosen_speaker_two_prefix: str = "Speaker 2"
):
    """
    Given a text prompt, generate audio using Google TTS
    """
    response = tts_client.models.generate_content(
    model=model_name,
    contents=text,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker=chosen_speaker_one_prefix,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=speaker_one_voice,
                        )
                    )
                ),
                types.SpeakerVoiceConfig(
                    speaker=chosen_speaker_two_prefix,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=speaker_two_voice,
                        )
                    )
                ),
                ]
            )
        )
    )
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    write_audio_data_to_wav_file(output_file, data)


def write_audio_data_to_wav_file(
        output_path: Path, 
        audio_data, 
        channels=1, 
        rate=24000, 
        sample_width=2
):
   with wave.open(str(output_path), "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(audio_data)


def combine_wav_files(
        input_files: List[Path], 
        output_file: Path,
        silence_duration_sec: float = 0.4  # 0.4 seconds
):
    """
    Combines a list of WAV files into a single WAV file.
    This function assumes all input files have the same
    sample rate, bit depth, and number of channels.
    """
    # Read the first file to get its parameters
    first_file = wave.open(str(input_files[0]), 'rb')
    params = first_file.getparams()
    first_file.close()

    n_channels, sample_width, frame_rate = params[:3]

    # Calculate silence frames and data
    silence_frames = int(frame_rate * silence_duration_sec)
    frame_size = n_channels * sample_width
    silent_data = b'\x00' * (silence_frames * frame_size)

    # Create a new WAV file for writing
    output_wave = wave.open(str(output_file), 'wb')
    output_wave.setparams(params)

    # Loop through each input file, read its data, and write to the output
    for ix, file_path in enumerate(input_files):
        with wave.open(str(file_path), 'rb') as input_wave:
            # Read all audio frames from the current file
            frames = input_wave.readframes(input_wave.getnframes())

            # Write the frames to the output file
            output_wave.writeframes(frames)

        # Add silence after every file, except the last one
        if ix < len(input_files) - 1:
            output_wave.writeframes(silent_data)

    # Close the output file
    output_wave.close()


def delete_files(
    files_to_delete: List[Path]
):
    """
    Given a list of filepaths,
    delete the files.
    """
    for file_path in files_to_delete:
        try:
            # Check if the path is a file and then delete it
            if file_path.is_file():
                file_path.unlink()
                typer.echo(f"Successfully deleted {file_path}")
            else:
                typer.echo(f"Skipping {file_path}: It is not a file.")
        except FileNotFoundError:
            typer.echo(f"Error: {file_path} not found.")
        except Exception as e:
            typer.echo(f"An error occurred while deleting {file_path}: {e}")


def check_api_key(
        service_name: str,
        username: str,
        force_prompt: bool = False
    ) -> str:
    """
    Retrieve API key from keyring, prompt user if not 
    found or force_prompt is True.
    """
    api_key = None

    if not force_prompt:
        api_key = keyring.get_password(service_name, username)

    if not api_key or force_prompt:
        typer.echo(f"{username} API key is not set or invalid.")
        api_key = typer.prompt(
            f"Please enter your {username} API key",
            hide_input=False,
            confirmation_prompt=True,
        )
        keyring.set_password(service_name, username, api_key)
        typer.echo("API key securely saved.")

    return api_key


def validate_voices_elevenlabs(
    client: ElevenLabs,
    speaker_one_voice: str,
    speaker_two_voice: str
):
    """
    Validate user/default selections against
    list retrieved from ElevenLabs API

    Note that we won't be able to use community voices
    until they've been added to our account.
    """
    voice_one_invalid = False
    voice_two_invalid = False

    try:
        voice_one_result = client.voices.get(voice_id=speaker_one_voice)
    except:
        typer.echo(f"Voice selection one ({speaker_one_voice}) is invalid.")
        voice_one_invalid = True
    try:
        voice_two_result = client.voices.get(voice_id=speaker_two_voice)
    except:
        typer.echo(f"Voice selection two ({speaker_two_voice}) is invalid.")
        voice_two_invalid = True
    if voice_one_invalid or voice_two_invalid:
        typer.echo("Please rerun the script with valid voice selections.")
        raise typer.Exit(1)


def select_voices_google(
    speaker_one_voice: Optional[str],
    speaker_two_voice: Optional[str]
):
    """
    Select Google TTS voices, based on user input and defaults.
    """
    voices_left_to_choose_from = list(VOICES_GOOGLE)
    
    # remove selected voices from voice candidates list, as applicable
    if speaker_one_voice:
        try:
            matching_index = voices_left_to_choose_from.index(speaker_one_voice)
            voices_left_to_choose_from.pop(matching_index)
        except:
            typer.echo(f"Invalid voice ({speaker_one_voice}) selected for speaker one. Exiting.")
            raise typer.Exit(1)
    if speaker_two_voice:
        try:
            matching_index = voices_left_to_choose_from.index(speaker_two_voice)
            voices_left_to_choose_from.pop(matching_index)
        except:
            typer.echo(f"Invalid voice ({speaker_two_voice}) selected for speaker two. Exiting.")
            raise typer.Exit(1)

    # set remaining voices, as applicable
    if not speaker_one_voice:
        random_index = random.randint(0, len(voices_left_to_choose_from) - 1)
        speaker_one_voice = voices_left_to_choose_from.pop(random_index)
    if not speaker_two_voice:
        random_index = random.randint(0, len(voices_left_to_choose_from) - 1)
        speaker_two_voice = voices_left_to_choose_from.pop(random_index)    
    
    return speaker_one_voice, speaker_two_voice


def clean_and_validate_speaker_labels(
    speaker_one_prefix: Optional[str],
    speaker_two_prefix: Optional[str]
):
    """
    Clean and validate speaker labels for/in transcript 
    and for use in multivoice TTS, based on user input and defaults.
    """
    if speaker_one_prefix:
        speaker_one_prefix = speaker_one_prefix.strip()
        if speaker_one_prefix[-1] == ":":
            chosen_speaker_one_prefix = speaker_one_prefix[:-1]
        else:
            chosen_speaker_one_prefix = speaker_one_prefix
    else:
        chosen_speaker_one_prefix = DEFAULT_SPEAKER_ONE_LABEL

    if speaker_two_prefix:
        speaker_two_prefix = speaker_two_prefix.strip()
        if speaker_two_prefix[-1] == ":":
            chosen_speaker_two_prefix = speaker_two_prefix[:-1]
        else:
            chosen_speaker_two_prefix = speaker_two_prefix
    else:
        chosen_speaker_two_prefix = DEFAULT_SPEAKER_TWO_LABEL

    if chosen_speaker_one_prefix == chosen_speaker_two_prefix:
        typer.echo(
                f"Speaker prefixes/labels must be unique. "
                f"Speaker 1 label: {chosen_speaker_one_prefix}, "
                f"Speaker 2 label: {chosen_speaker_two_prefix}. "
                f"Exiting."
            )
        raise typer.Exit(1)

    return chosen_speaker_one_prefix, chosen_speaker_two_prefix


def execute_pdf_workflow(
    path_to_pdf: Path,
    output_dir: Path,
    api_key: str,
    timestamp: int,
    text_model: str = 'gemini-2.5-flash'
):
    """
    Workflow for performing document-understanding inference
    when PDF is passed in for summarization.
    """
    if not file_is_valid(path_to_pdf, '.pdf', 20):
        typer.echo("Exiting...")
        raise typer.Exit(code=1)        
    typer.echo("Generating text summary from PDF. This may take a few minutes...")	    
    text_summary = infer_with_pdf_document_understanding(path_to_pdf, TEXT_SUMMARY_PROMPT, api_key, text_model)

    if text_summary:
        text_summary_output_path = Path(output_dir / f'text_summary_{timestamp}.txt')
        write_text_to_file(text_summary, text_summary_output_path)
        typer.echo(f"Text summary written to {text_summary_output_path}")
        typer.confirm(
            (
                "Do you wish to convert this summary to transcript?" 
                "To edit the summary first, choose No to exit, edit the summary file, "
                "then rerun the script using the `--text_summary_file` flag."
            ),
            False,
            True,
        )
        return text_summary

def add_speaker_labels_to_transcript(
    transcript: str,
    speaker_one_prefix: str,
    speaker_two_prefix: str
):
    """
    Add speaker labels (Speaker 1, Speaker 2) to 
    conversation turns in transcript.

    Assumes transcript conversation turns are
    newline delimited.
    """
    labeled_statements = []
    original_statements = [line for line in transcript.splitlines() if line]
    for ix, statement in enumerate(original_statements):
        if ix == 0 or ix % 2 == 0:
            statement = f"{speaker_one_prefix}: {statement}"
        else:
            statement = f"{speaker_two_prefix}: {statement}"
        labeled_statements.append(statement)
    labeled_transcript = '\n\n'.join(labeled_statements)
    return labeled_transcript

def execute_transcript_generation_workflow(
    text_summary: str,
    output_dir: Path,
    api_key: str,
    sys_instrux: str,
    timestamp: int,
    speaker_one_prefix: Optional[str] = None,
    speaker_two_prefix: Optional[str] = None,
    text_model: str = 'gemini-2.5-flash',
):
    """
    Workflow for generating a podcast transcript from
    text summary.
    """
    typer.echo("Generating transcript from text summary. This may take a few minutes...")
    transcript = generate_text(text_summary, api_key, sys_instrux, text_model)
    
    if speaker_one_prefix and speaker_two_prefix:
        transcript = add_speaker_labels_to_transcript(transcript, speaker_one_prefix, speaker_two_prefix)
    
    transcript_output_path = Path(output_dir / f'transcript_{timestamp}.txt')
    write_text_to_file(transcript, transcript_output_path)
    typer.echo(f"Transcript written to {transcript_output_path}")
    typer.confirm(
        (
            "Do you wish to convert this transcript to audio?" 
            "To edit the transcript first (which is highly recommended!), "
            "choose No to exit, edit the transcript file, "
            "then rerun the script using the `--transcript_file` flag."
        ),
        False,
        True,
    )
    return transcript


if __name__ == "__main__":
    app()
