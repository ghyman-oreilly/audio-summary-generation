from concurrent.futures import ThreadPoolExecutor, TimeoutError
from elevenlabs import ElevenLabs, VoiceSettings
from google import genai
from google.genai import types
import json
import keyring
import nltk
from pathlib import Path
from simple_term_menu import TerminalMenu
import time
import typer
from typing import List, Optional, Union
import wave

from prompts import TEXT_SUMMARY_PROMPT, TRANSCRIPT_SYS_INSTRUCTIONS


def write_backup_to_json_file(
    input_data: list[dict], output_filepath: Union[str, Path]
):
    """
    Save backup data to JSON file.

    These backups allow the user to more easily regenerate
    audio from segments of the transcript.
    """
    with open(str(output_filepath), "w") as f:
        json.dump(input_data, f)

def read_backup_from_json_file(input_filepath: Union[str, Path]):
    """
    Read JSON file and and validate data.
    """
    with open(str(input_filepath), "r") as f:
        data = json.load(f)
        validate_backup_data(data)
        return data

def validate_backup_data(
    json_data: List[dict],
    expected_fields: List[str] = ['voice_id', 'text', 'filepath', 'request_id']
):
    """
    Validate that loaded backup data matches
    expected shape and fields.
    """
    is_valid = True
    
    if not isinstance(json_data, list):
        is_valid = False

    for item in json_data:
        if not isinstance(item, dict):
            is_valid = False     
        
        for field in expected_fields:
            try:
                field_value = item.get(field)
                if not isinstance(field_value, str):
                    is_valid = False
            except:
                is_valid = False
    
    if not is_valid:
        typer.echo(
            f'Backup data must comprise a JSON list of objects, '
            f'each of which should have the following fields with '
            f'string values: {expected_fields}\n'
            f'Exiting.'
        )
        typer.Exit(1)
    
    return is_valid

def generate_menu(options, title=None, **kwargs):
    terminal_menu = TerminalMenu(options, title=title, **kwargs)
    index = terminal_menu.show()
    return index

def check_tokenizer_data_availability():
    """
    Check for availability of sentence tokenizer
    data. Download if not available.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        typer.echo(
            (
                "Downloading NLTK 'punkt' tokenizer data. "
                "This is a one-time process requiring internet access."
            )
        )
        nltk.download('punkt')

app = typer.Typer()

check_tokenizer_data_availability()

# https://elevenlabs.io/app/default-voices
# users can indicate other voices they want to use,
# but nondefault voices must be added to the account 
# collection first
DEFAULT_VOICE_ONE_ELEVENLABS = 'SAz9YHcvj6GT2YYXdXww' # River
DEFAULT_VOICE_TWO_ELEVENLABS = 'bIHbv24MWmeRgasZH58o' # Will

# keychain deets
SERVICE_NAME = "audio_summary_generator"
GEMINI_KEY_USER_NAME = "google_api_key"
ELEVENLABS_KEY_USER_NAME = "elevenlabs_api_key"

@app.command(
    help="""
    Combine audio chunk files from previous
    session(s) into a single audio file. 
    """
)
def combine_audio_files(
    input_files: List[Path],
    output_dir: Optional[Path] = typer.Option(
    None, help=(
            "Provide directory where combine output audio file should be saved. "
            "Defaults to current working directory."
        )
    ),
):
    timestamp = int(time.time())
    
    # validate input files
    if not input_files or len(input_files) < 2:
        typer.echo("Two or more input files are required. Exiting...")
        raise typer.Exit(code=1) 
    
    for input_file in input_files:
        if not file_is_valid(input_file, '.wav', 20):
            typer.echo(f"Input file {input_file} isn't valid. Exiting...")
            raise typer.Exit(code=1) 
    
    # check/config output directory
    if output_dir:
        if not dir_is_valid(output_dir):
            typer.echo("Output directory isn't valid. Exiting...")
            raise typer.Exit(code=1)
    else:
        output_dir = Path.cwd()

    combined_audio_filepath = Path(output_dir / f"combined_audio_{timestamp}.wav")
    combine_wav_files(input_files, combined_audio_filepath)
    typer.echo(f"Combined audio saved to {str(combined_audio_filepath)}")

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
            )
        ),
        backup_file_for_regen: Optional[Path] = typer.Option(
            None, help=(
                "Provide path to JSON file containing backup data. "
                "This will provide an opportunity to regenerate previously generated segments. "
                "If providing this text file, the PDF path argument and text_summary_file options "
                "don't need to be provided. Any output_filepath argument will be ignored, as the "
                "directory containing the previously generated audio files will be used."
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
        GEMINI_API_KEY = check_api_key(SERVICE_NAME, GEMINI_KEY_USER_NAME)
        ELEVENLABS_API_KEY = check_api_key(SERVICE_NAME, ELEVENLABS_KEY_USER_NAME)

        tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        validate_voices_elevenlabs(tts_client, speaker_one_voice, speaker_two_voice)

        TEXT_MODEL = 'gemini-2.5-pro'
        TTS_MODEL = 'eleven_multilingual_v2'

        timestamp = int(time.time())
        
        if (
            not path_to_pdf 
            and not text_summary_file 
            and not transcript_file
            and not backup_file_for_regen
        ):
                typer.echo(
                    "path_to_pdf required if not providing an existing "
                    "text summary, transcript, or backup file. Exiting."
                )
                raise typer.Exit(1)

        # check/config output directory
        if output_dir and not backup_file_for_regen:
            if not dir_is_valid(output_dir):
                typer.echo("Output directory isn't valid. Exiting...")
                raise typer.Exit(code=1)
        else:
            output_dir = Path.cwd()

        backup_filepath = Path(output_dir / f"backup_{timestamp}.json")

        # generate text summary
        if (
            path_to_pdf 
            and not text_summary_file 
            and not transcript_file
            and not backup_file_for_regen
        ):
            text_summary = execute_pdf_workflow(path_to_pdf, output_dir, GEMINI_API_KEY, timestamp, TEXT_MODEL)

        # handle existing/inputted text summary
        if (
            text_summary_file
            and not transcript_file
            and not backup_file_for_regen
        ):
            if not file_is_valid(text_summary_file, '.txt'):
                typer.echo("Exiting...")
                raise typer.Exit(code=1)
            text_summary = read_text_from_file(text_summary_file)
        
        # generate transcript
        if not transcript_file and not backup_file_for_regen:
            transcript = execute_transcript_generation_workflow(
                text_summary,
                output_dir,
                GEMINI_API_KEY,
                TRANSCRIPT_SYS_INSTRUCTIONS,
                timestamp,
                TEXT_MODEL
            )

        # handle existing/inputted transcript
        if transcript_file and not backup_file_for_regen:
            if not file_is_valid(transcript_file, '.txt'):
                typer.echo("Exiting...")
                raise typer.Exit(code=1)
            transcript = read_text_from_file(transcript_file)

        if not backup_file_for_regen:
            # chunk transcript
            typer.echo("Chunking transcript. This may take a few minutes...")
            transcript_chunks = create_speaker_text_chunks(transcript)
            typer.echo(f"Transcript split into {len(transcript_chunks)} chunks.")

            # create generation plan
            # TODO: abstract
            generation_data = []
            audio_chunk_filepaths = []
            for ix, transcript_chunk in enumerate(transcript_chunks):
                output_filepath = Path(output_dir / f"audio_chunk_{ix:03d}_{timestamp}.wav")
                if ix == 0 or ix % 2 == 0:
                    voice_id = speaker_one_voice
                else:
                    voice_id = speaker_two_voice
                for text_string in transcript_chunk: 
                    generation_data.append(
                        {
                            'voice_id': voice_id, 
                            'text': text_string, 
                            'filepath': str(output_filepath)
                        }
                    )
                    audio_chunk_filepaths.append(output_filepath)
            
            # generate audio
            # iterate over generation data
            typer.echo(
                "Generating audio from transcript chunks. "
                "This could take a while (up to 10 minutes per chunk)..."
            )
            # TODO: abstract
            request_ids = []
            for ix, generation_datum in enumerate(generation_data):
                text_string = generation_datum["text"]
                voice_id = generation_datum["voice_id"]
                output_filepath = Path(generation_datum["filepath"])
                typer.echo(f"Generating audio chunk {ix+1} of {len(generation_data)}...")
                request_id = generate_audio_with_timeout(
                    text=text_string,
                    voice_id=voice_id,
                    output_file=output_filepath,
                    tts_client=tts_client,
                    model_id=TTS_MODEL,
                    previous_request_ids=request_ids[-1:] # previous 1 ID (we can include up to 3,
                                                          # but for our use case, 1 seems optimal, based 
                                                          # on some experimentation)
                )
                request_ids.append(request_id)
                generation_data[ix]['request_id'] = request_id
            write_backup_to_json_file(generation_data, backup_filepath)
        else:
            # read and validate saved generation plan
            if not file_is_valid(backup_file_for_regen, '.json'):
                typer.echo("Exiting...")
                raise typer.Exit(code=1)

            all_generation_data = read_backup_from_json_file(backup_file_for_regen)

            # set output directory
            output_dir = Path(all_generation_data[1]['filepath']).parent

            # TODO: validate voice IDs?
            # TODO: validate segment filepaths

            # user selects segments to regen
            ix_of_items_to_regen = generate_menu(
                [Path(x.get('filepath')).name for x in all_generation_data],
                'Select audio segments to regenerate',
                multi_select=True
            )

            if ix_of_items_to_regen is None:
                typer.echo('Exiting.')
                raise typer.Exit(0)

            data_to_regenerate = [(i, all_generation_data[i]) for i in ix_of_items_to_regen]
            new_filepaths_lookup_map = {}

            # regen segments
            for ix, generation_datum_tuple in enumerate(data_to_regenerate):
                original_segment_ix, generation_datum = generation_datum_tuple
                text_string = generation_datum["text"]
                voice_id = generation_datum["voice_id"]
                try:
                    previous_request_id = data_to_regenerate[original_segment_ix-1] if original_segment_ix != 0 else ''
                except:
                    previous_request_id = ''
                output_filepath = Path(output_dir / f"audio_chunk_{original_segment_ix:03d}_{timestamp}.wav")
                typer.echo(f"Generating audio chunk {ix+1} of {len(data_to_regenerate)}...")
                generate_audio_with_timeout(
                    text=text_string,
                    voice_id=voice_id,
                    output_file=output_filepath,
                    tts_client=tts_client,
                    model_id=TTS_MODEL,
                    previous_request_ids=[previous_request_id]
                )
                new_filepaths_lookup_map[original_segment_ix] = str(output_filepath)
            
            # collect updated and original segment filepaths
            audio_chunk_filepaths = [
                new_filepaths_lookup_map.get(i, x['filepath']) 
                for i, x in enumerate(all_generation_data)
            ]

        # combine chunk audio files
        typer.echo("Combining audio chunk files...")
        combined_audio_filepath = Path(output_dir / f"combined_audio_{timestamp}.wav")
        combine_wav_files(audio_chunk_filepaths, combined_audio_filepath)
        typer.echo(f"Combined audio saved to {str(combined_audio_filepath)}")

        save_audio_chunks = typer.confirm(
            (
                "Do you wish to save the partial audio chunks?\n "
                "Choose Yes (default) if you wish to save them, in case "
                "they some segments must be regenerated later. "
                "Otherwise, select No to delete the segments."
            ),
            default=True
        )

        if not save_audio_chunks:
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
        model_name: str = 'gemini-2.5-pro'
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
        model_name: str = 'gemini-2.5-pro'
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
            model=model_name,
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
      

def create_speaker_text_chunks(
    text_string: str,
    char_limit: int = 3000
):
    """
    Generate a list of strings from a single string,
    splitting by lines. If a line exceeds char_limit, it's 
    further split by sentences using the helper function.

    Returns list of lists of strings. Each sublist represents
    the speaking turn of a speaker.
    """  
    chunks_by_speaker = []
    # Filter out empty lines
    lines = [line for line in text_string.splitlines() if line]
    
    for line in lines:
        # Check if the line (speaker segment) is within the limit
        if len(line) <= char_limit:
            # If within limit, append as a single-item list
            chunks_by_speaker.append([line])
        else:
            # If over limit, use the helper function to split by sentences
            sub_chunks = chunk_segment_by_sentences(line, char_limit)
            # Append the resulting list of smaller chunks
            chunks_by_speaker.append(sub_chunks)
            
    return chunks_by_speaker

    
def chunk_segment_by_sentences(
    segment: str,
    char_limit: int
) -> List[str]:
    """
    Chunks a single speaker segment into smaller strings
    based on sentences, ensuring each chunk respects the char_limit.
    This is only called when the initial segment exceeds char_limit.
    """
       
    # Use NLTK to split the content into sentences
    sentences = nltk.tokenize.sent_tokenize(segment)
    
    current_speaker_chunks = []
    current_chunk_content = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Prepare for length calculation
        separator = " " if current_chunk_content else ""
        
        # Calculate the length of the new chunk *if* the sentence is added
        total_len_if_added = len(current_chunk_content) + len(separator) + len(sentence)

        # Individual sentence exceeds the limit
        if len(sentence) > char_limit:
            typer.echo(
                    (
                        f"Warning: Sentence in segment '{segment[:50]}...' "
                        f"exceeds char_limit ({char_limit}). Please edit and rerun "
                        f"using the --transcript_file flag."
                    )
                )
            raise typer.Exit(code=1)
        
        # Adding the new sentence would exceed the limit
        if total_len_if_added > char_limit:
            # Flush the current accumulated chunk
            if current_chunk_content:
                current_speaker_chunks.append(current_chunk_content)
            
            # Start a new chunk with the current sentence
            current_chunk_content = sentence
        else:
            # Otherwise, append the sentence to the current chunk content
            current_chunk_content += separator + sentence

    # Append any remaining text
    if current_chunk_content:
        current_speaker_chunks.append(current_chunk_content)
            
    return current_speaker_chunks


def generate_audio_with_timeout(
    text: str,
    voice_id: str,
    output_file: Path,
    tts_client: ElevenLabs,
    model_id: str = 'eleven_multilingual_v2',
    timeout: float = (60.0 * 10), # 10 minutes per chunk/call
    previous_request_ids: list[str] = []
):
    """
    Wrapper around generate_audio_chunk_from_chunk,
    designed to allow for custom timeout.
    """
    # use ThreadPoolExecutor to allow for custom timeout on execution of API calls
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(
                generate_audio_chunk_from_chunk,
                text=text, 
                voice_id=voice_id,
                output_file=output_file,
                tts_client=tts_client,
                model_id=model_id,
                previous_request_ids=previous_request_ids
            )

            # wait for the executor result, enforcing timeout
            request_id = future.result(timeout=timeout)

            typer.echo(f"Audio chunk saved to {str(output_file)}...")

            return request_id
        except TimeoutError:
            typer.echo(f"API call exceeded timeout of {timeout / 60} minutes. Exiting.")
            raise typer.Exit(code=1)

def generate_audio_chunk_from_chunk(
    text: str,
    voice_id: str,
    output_file: Path,
    tts_client: ElevenLabs,
    model_id: str = 'eleven_multilingual_v2',
    previous_request_ids: list[str] = []
):
    """
    Given a text string, generate audio using ElevenLabs
    Text to Speech API
    """
    # TODO: check previous_request_ids, next_request_ids params of convert!
    
    # max of three previous request IDs are accepted
    # https://elevenlabs.io/docs/cookbooks/text-to-speech/request-stitching
    if previous_request_ids:
        previous_request_ids = previous_request_ids[-3:]

    with tts_client.text_to_speech.with_raw_response.convert(
        model_id=model_id,
        text=text,
        voice_id=voice_id,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0
            ),
        previous_request_ids=previous_request_ids,
        output_format='pcm_24000' # important to use this encoding
                                  # for compatibility with write_audio_data_to_wav_file
    ) as response:
        request_id = response._response.headers.get("request-id")
        audio_data = b''.join(chunk for chunk in response.data)
    
    write_audio_data_to_wav_file(output_file, audio_data)

    return request_id


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
    files_to_delete: List[Union[Path, str]]
):
    """
    Given a list of filepaths,
    delete the files.
    """
    for file_path in files_to_delete:
        file_path = Path(file_path)
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

def execute_pdf_workflow(
    path_to_pdf: Path,
    output_dir: Path,
    api_key: str,
    timestamp: int,
    text_model: str = 'gemini-2.5-pro'
):
    """
    Workflow for performing document-understanding inference
    when PDF is passed in for summarization.
    """
    if not file_is_valid(path_to_pdf, '.pdf', 20):
        typer.echo("Input file isn't valid. Exiting...")
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
    sys_instrux: str,
    timestamp: int,
    text_model: str = 'gemini-2.5-pro',
):
    """
    Workflow for generating a podcast transcript from
    text summary.
    """
    typer.echo("Generating transcript from text summary. This may take a few minutes...")
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





if __name__ == "__main__":
    app()
