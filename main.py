from google import genai
from google.genai import types
import keyring
from pathlib import Path
import random
import time
import typer
from typing import List, Optional
import wave

from prompts import TEXT_SUMMARY_PROMPT, TRANSCRIPT_SYS_INSTRUCTIONS


# https://ai.google.dev/gemini-api/docs/speech-generation
VOICES = [
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

DEFAULT_SPEAKER_ONE_LABEL = 'Speaker 1'
DEFAULT_SPEAKER_TWO_LABEL = 'Speaker 2'

SERVICE_NAME = "audio_summary_generator"
USERNAME = "google_api_key"

def main(
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
                f"Available options are: {', '.join(VOICES)} "
                f"Default: random."
            )
        ),
        speaker_two_voice: Optional[str] = typer.Option(
            None,
            help=(
                f"Choose a voice for speaker one. "
                f"Available options are: {', '.join(VOICES)} "
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
    API_KEY = check_api_key()

    TEXT_MODEL = 'gemini-2.5-flash'
    TTS_MODEL = 'gemini-2.5-flash-preview-tts'

    TIMESTAMP = int(time.time())
    
    # check/config output directory
    if output_dir:
        if not dir_is_valid(output_dir):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
    else:
        output_dir = Path.cwd()

    speaker_one_voice, speaker_two_voice = select_voices(speaker_one_voice, speaker_two_voice)

    chosen_speaker_one_prefix, chosen_speaker_two_prefix = select_speaker_labels(
        speaker_one_prefix, speaker_two_prefix
    )

    # generate text summary
    if path_to_pdf and not text_summary_file and not transcript_file:
        text_summary = execute_pdf_workflow(path_to_pdf, output_dir, API_KEY, TIMESTAMP, TEXT_MODEL)

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
            API_KEY,
            TRANSCRIPT_SYS_INSTRUCTIONS,
            TIMESTAMP,
            chosen_speaker_one_prefix,
            chosen_speaker_two_prefix,
            TEXT_MODEL
        )

    # handle existing/inputted transcript
    if transcript_file:
        if not file_is_valid(transcript_file, '.txt'):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
        transcript = read_text_from_file(transcript_file)

    # validate transcript
    if not transcript_validates(
        transcript,
        chosen_speaker_one_prefix,
        chosen_speaker_two_prefix
    ):
        typer.echo(
            f"Selected speaker prefixes/labels not found in transcript. "
            f"Please check and rerun script. "
            f"Selected speaker 1 label: {chosen_speaker_one_prefix}, "
            f"selected speaker 2 label: {chosen_speaker_two_prefix}. "
            f"Exiting."
        )
        raise typer.Exit(1)

    # chunk transcript
    typer.echo("Chunking transcript. This may take a few minutes...")
    transcript_chunks = chunk_string(transcript, API_KEY, TTS_MODEL)
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
        API_KEY,
        speaker_one_voice=speaker_one_voice,
        speaker_two_voice=speaker_two_voice,
        chosen_speaker_one_prefix=chosen_speaker_one_prefix,
        chosen_speaker_two_prefix=chosen_speaker_two_prefix  
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


def chunk_string(
    text_string: str,
    api_key: str,
    model_name: str = 'gemini-2.5-flash-preview-tts',
    token_limit: Optional[int] = None
):
    """
    Generate a list of strings from a single string,
    keeping within a specified token limit.
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
    text_chunks: List[str],
    timestamp: int,
    output_dir: Path,
    api_key: str,
    model_name: str = 'gemini-2.5-flash-preview-tts',
    speaker_one_voice: str = 'Puck',
    speaker_two_voice: str = 'Zephyr',
    chosen_speaker_one_prefix: str = "Speaker 1",
    chosen_speaker_two_prefix: str = "Speaker 2"  
):
    """
    Given a list of text chunks, generate
    and save audio chunks to file, returning
    list of audio chunk filepaths.
    """
    audio_chunk_filepaths = []

    for i, text_chunk in enumerate(text_chunks):
        typer.echo(f"Generating audio chunk {i+1} of {len(text_chunks)}...")
        audio_chunk_filepath = Path(output_dir / f"audio_chunk_{i:03d}_{timestamp}.wav")
        generate_audio_chunk_from_text_chunk(
            text_chunk, 
            audio_chunk_filepath, 
            api_key=api_key, 
            model_name=model_name,
            speaker_one_voice=speaker_one_voice,
            speaker_two_voice=speaker_two_voice,
            chosen_speaker_one_prefix=chosen_speaker_one_prefix,
            chosen_speaker_two_prefix=chosen_speaker_two_prefix
        )
        audio_chunk_filepaths.append(audio_chunk_filepath)
        typer.echo(f"Audio chunk saved to {str(audio_chunk_filepath)}...")
    
    return audio_chunk_filepaths


def generate_audio_chunk_from_text_chunk(
    text: str,
    output_file: Path,
    api_key: str,
    model_name: str = 'gemini-2.5-flash-preview-tts',
    speaker_one_voice: str = 'Puck',
    speaker_two_voice: str = 'Zephyr',
    chosen_speaker_one_prefix: str = "Speaker 1",
    chosen_speaker_two_prefix: str = "Speaker 2"
):
    """
    Given a text prompt, generate audio
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
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
        output_file: Path
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

    # Create a new WAV file for writing
    output_wave = wave.open(str(output_file), 'wb')
    output_wave.setparams(params)

    # Loop through each input file, read its data, and write to the output
    for file_path in input_files:
        with wave.open(str(file_path), 'rb') as input_wave:
            # Read all audio frames from the current file
            frames = input_wave.readframes(input_wave.getnframes())

            # Write the frames to the output file
            output_wave.writeframes(frames)

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


def check_api_key(force_prompt: bool = False) -> str:
    """
    Retrieve Gemini API key from keyring, prompt user if not 
    found or force_prompt is True.
    """
    api_key = None

    if not force_prompt:
        api_key = keyring.get_password(SERVICE_NAME, USERNAME)

    if not api_key or force_prompt:
        typer.echo("Gemini API key is not set or invalid.")
        api_key = typer.prompt(
            "Please enter your Gemini API key",
            hide_input=False,
            confirmation_prompt=True,
        )
        keyring.set_password(SERVICE_NAME, USERNAME, api_key)
        typer.echo("API key securely saved.")

    return api_key


def select_voices(
    speaker_one_voice: Optional[str],
    speaker_two_voice: Optional[str]
):
    """
    Select TTS voices, based on user input and defaults.
    """
    voices_left_to_choose_from = list(VOICES)
    
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


def select_speaker_labels(
    speaker_one_prefix: Optional[str],
    speaker_two_prefix: Optional[str]
):
    """
    Select labels for/in transcript and for use in multvoice TTS, 
    based on user input and defaults.
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


def execute_transcript_generation_workflow(
    text_summary: str,
    output_dir: Path,
    api_key: str,
    unformatted_sys_instrux: str,
    timestamp: int,
    chosen_speaker_one_prefix: str,
    chosen_speaker_two_prefix: str,
    text_model: str = 'gemini-2.5-flash',
):
    """
    Workflow for generating a podcast transcript from
    text summary.
    """
    typer.echo("Generating transcript from text summary. This may take a few minutes...")
    sys_instrux = format_sys_instrux(
        unformatted_sys_instrux, 
        chosen_speaker_one_prefix,
        chosen_speaker_two_prefix
    )
    transcript = generate_text(text_summary, api_key, sys_instrux, text_model)
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

def format_sys_instrux(
    unformatted_sys_instrux: str,
    chosen_speaker_one_prefix: str,
    chosen_speaker_two_prefix: str,
) -> str:
    """
    Format sys instrux template,
    using the chosen speaker prefixes
    to replace the placeholders.
    """
    if validate_sys_instrux_format:   
        sys_instrux = unformatted_sys_instrux.format(
        speaker_1=chosen_speaker_one_prefix, 
        speaker_2=chosen_speaker_two_prefix
        )
        return sys_instrux
    else:
        typer.echo(
            "`speaker_1` and `speaker_2` fields not found in "
            "`unformatted_sys_instrux` template. Exiting."
        )
        raise typer.Exit(1)

def validate_sys_instrux_format(
    unformatted_sys_instrux: str
) -> bool:
    """
    Check that unformatted sys instrux
    template contains the expected fields.

    We can probably make this more elegant
    if we find we're adding fields over time.
    """
    if (
        not '{speaker_1}' in unformatted_sys_instrux
        or not not '{speaker_2}' in unformatted_sys_instrux
    ):
        return False
    return True

def transcript_validates(
    transcript: str,
    chosen_speaker_one_prefix: str,
    chosen_speaker_two_prefix: str
):
    """
    Make sure transcript has the expected
    speaker labels
    """
    if (
        not chosen_speaker_one_prefix in transcript
        or not chosen_speaker_two_prefix in transcript
    ):
        return False
    return True


if __name__ == "__main__":
    typer.run(main)
