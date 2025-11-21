from concurrent.futures import ThreadPoolExecutor, TimeoutError
from elevenlabs import ElevenLabs, VoiceSettings
from google import genai
from google.genai import types
import json
import keyring
import nltk
from pathlib import Path
import random
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

def read_backup_from_json_file(
    input_filepath: Union[str, Path],
    tts_client: Optional[ElevenLabs] = None,
    validate_voices: bool = True,
    expected_fields: List[str] = ['voice_id', 'text', 'filepath', 'request_id']
):
    """
    Read JSON file and and validate data.
    """
    with open(str(input_filepath), "r") as f:
        data = json.load(f)
        validate_backup_data(data, tts_client, validate_voices, expected_fields)
        return data

def validate_backup_data(
        data: List[dict],
        tts_client: Optional[ElevenLabs] = None,
        validate_voices: bool = True,
        expected_fields: List[str] = ['voice_id', 'text', 'filepath', 'request_id']
    ):
    """
    Wrapper for backup data validation steps
    """
    validate_backup_data_shape(data, expected_fields)
    if validate_voices:
        validate_backup_data_voice_ids(data, tts_client)
    validate_backup_data_segment_filepaths(data)

def validate_backup_data_voice_ids(
        data: List[dict],
        tts_client: ElevenLabs
    ):
    """
    Validate voice IDs in backup data
    """
    voice_ids = list(set([x['voice_id'] for x in data]))

    if len(voice_ids) != 2:
        typer.echo(
            (
                "Backup data must contain exactly two unique voice IDs "
                "(duplicates are okay). Exiting."
            )
        )
        raise typer.Exit(1)

    validate_voices(tts_client, voice_ids[0], voice_ids[1])


def validate_backup_data_segment_filepaths(data: List[dict]):
    """
    Validate filepaths in backup data
    """
    filepaths = [Path(x['filepath']) for x in data]
    invalid_filepaths = [str(x) for x in filepaths if not file_is_valid(x, '.wav')]
    invalid_filepaths_str = '\n'.join(invalid_filepaths)

    if invalid_filepaths:
        typer.echo(
            (
                f"One or more filepaths in backup data is invalid:\n"
                f"\n{invalid_filepaths_str}\n\nExiting."
             )
        )
        raise typer.Exit(1)

def validate_backup_data_shape(
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
        raise typer.Exit(1)
    
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
    input_files: Optional[List[Path]] = typer.Argument(
        None, help=(
            "Provide comma-delimited list of segment WAV files "
            "to combine. Argument not required if a valid backup "
            "JSON file is provided using the XX option."
        )
    ),
    output_dir: Optional[Path] = typer.Option(
    None, help=(
            "Provide directory where combine output audio file should be saved. "
            "Defaults to current working directory."
        )
    ),
    backup_file: Optional[Path] = typer.Option(
        None, help=(
            "Provide path to JSON file containing backup data. "
            "If providing this text file, the input_files argument "
            "doesn't need to be provided. Any output_filepath argument will be ignored, as the "
            "directory containing the previously generated audio files will be used."
        )
    ),
):
    timestamp = int(time.time())

    # input_files list use case    
    if not backup_file:
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
    # backup_file use case
    else:
        # validate backup file
        if not file_is_valid(backup_file, '.json'):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)
        
        # get filepaths
        backup_data = read_backup_from_json_file(
            backup_file, 
            validate_voices=False, 
            expected_fields=['filepath']
        )
        input_files = [x['filepath'] for x in backup_data]

        # set output directory
        output_dir = Path(input_files[1]).parent

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

    # validate CLI params
    if not voice_id or not custom_name:
        typer.echo(f"Voice ID and Custom Name params must have valid values. Exiting.")
        raise typer.Exit(code=1)

    # check for voice in user library
    if voice_exists_in_account_library(voice_id, client):
        typer.echo(f"Shared voice already exists in user library. Exiting.")
        raise typer.Exit(code=1) 

    # check for shared voice in community library
    # and get owner ID
    typer.echo(
        (
            "Searching for voice in community library. Please be patient, "
            "as this may take several minutes..."
        )
    )
    voice_owner_id = get_voice_owner_id_from_community_library(voice_id, client)

    if not voice_owner_id:
        typer.echo(f"Shared voice matching ID {voice_id} not found. Exiting.")
        raise typer.Exit(code=1)

    # add voice to user library
    client.voices.share(
        public_user_id=voice_owner_id,
        voice_id=voice_id,
        new_name=custom_name
    )

    typer.echo(f"Shared voice matching ID {voice_id} added to user library.")
    typer.echo(f"Script complete.")

def voice_exists_in_account_library(
    voice_id: str,
    client: ElevenLabs
):
    """
    Check whether voice (by ID) already
    exists in ElevenLabs account library.
    """
    try:
        client.voices.get(voice_id=voice_id)
        return True
    except Exception as e:
        if 'voice_not_found' in str(e):
            # expected result
            return False
        else:
            # Handle all other unexpected errors
            raise e

def get_voice_owner_id_from_community_library(
    voice_id: str,
    client: ElevenLabs
):
    """
    Check whether voice (by ID) exists
    in the ElevenLabs community library.

    Return owner ID (required for adding voice to user library) or None.
    """
    current_page = 0
    continue_paging = True
    while continue_paging:
        some_shared_voices = client.voices.get_shared(page_size=100, page=current_page)
        if some_shared_voices.voices:
            for voice in some_shared_voices.voices:
                if voice.voice_id == voice_id:
                    owner_id = voice.public_owner_id
                    return owner_id
            if not some_shared_voices.has_more:
                continue_paging = False
        current_page += 1
        time.sleep(0.5)
    return None

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
                f"Be sure to select a different voice than you chose for speaker one! "
                f"Default: {DEFAULT_VOICE_TWO_ELEVENLABS}."
            )
        ),
):
    _generate_audio_summary(
        path_to_pdf,
        output_dir,
        text_summary_file,
        transcript_file,
        backup_file_for_regen,
        speaker_one_voice,
        speaker_two_voice
    )
    
def _generate_audio_summary(
    path_to_pdf: Path,
    output_dir: Optional[Path] = None,
    text_summary_file: Optional[Path] = None,
    transcript_file: Optional[Path] = None,
    backup_file_for_regen: Optional[Path] = None,
    speaker_one_voice: str = DEFAULT_VOICE_TWO_ELEVENLABS,
    speaker_two_voice: str = DEFAULT_VOICE_TWO_ELEVENLABS
):
    """
    Script for generating an audio summary
    """
    GEMINI_API_KEY = check_api_key(SERVICE_NAME, GEMINI_KEY_USER_NAME)
    ELEVENLABS_API_KEY = check_api_key(SERVICE_NAME, ELEVENLABS_KEY_USER_NAME)

    ELEVENLABS_VOICE_SETTINGS = {
        'stability': 0.5,
        'similarity_boost': 0.75,
        'style': 0.0,
        'use_speaker_boost': True,
        'speed': 1.0
    }

    tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    voice_settings = generate_voice_settings(ELEVENLABS_VOICE_SETTINGS)
    validate_voices(tts_client, speaker_one_voice, speaker_two_voice)

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

    # audio generation flows
    if not backup_file_for_regen:
        # first-pass generation use case
        audio_chunk_filepaths = execute_audio_generation_workflow(
            transcript,
            output_dir,
            timestamp,
            speaker_one_voice,
            speaker_two_voice,
            tts_client,
            voice_settings,
            TTS_MODEL,
            backup_filepath
        )
    else:
        # segment(s) regeneration use case
        if not file_is_valid(backup_file_for_regen, '.json'):
            typer.echo("Exiting...")
            raise typer.Exit(code=1)

        output_dir, audio_chunk_filepaths = execute_audio_regeneration_workflow(
            backup_file_for_regen,
            tts_client,
            voice_settings,
            TTS_MODEL,
            timestamp
        )

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

def create_generation_data(
        transcript_chunks: list[str],
        output_dir: Path,
        timestamp: str,
        speaker_one_voice: str,
        speaker_two_voice: str   
):
    """
    From a list of transcript chunks,
    create a list of dicts with voice_id,
    text, and filepath fields.
    """
    generation_data = []
    string_counter = 0
    for ix, transcript_chunk in enumerate(transcript_chunks):
        if ix == 0 or ix % 2 == 0:
            voice_id = speaker_one_voice
        else:
            voice_id = speaker_two_voice
        for text_string in transcript_chunk: 
            output_filepath = Path(output_dir / f"audio_chunk_{string_counter:03d}_{timestamp}.wav")
            generation_data.append(
                {
                    'voice_id': voice_id, 
                    'text': text_string, 
                    'filepath': str(output_filepath)
                }
            )
            string_counter += 1
    return generation_data

def generate_audio_segments(
    generation_data: list[dict],
    tts_client: ElevenLabs,
    voice_settings: VoiceSettings,
    model_id: str,
    backup_filepath: Path
):
    """
    Given a generation_data list of dicts with
    text, voice_id, and filepath fields,
    iterate over the data, generating audio segments.

    Save audio segments to files.
    """
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
            voice_settings=voice_settings,
            model_id=model_id,
            previous_request_ids=request_ids[-1:] # previous 1 ID (we can include up to 3,
                                                # but for our use case, 1 seems optimal, based 
                                                # on some experimentation)
        )
        request_ids.append(request_id)
        generation_data[ix]['request_id'] = request_id
    write_backup_to_json_file(generation_data, backup_filepath)

def regenerate_audio_segments(
    data_to_regenerate: list[tuple[int, dict]],
    tts_client: ElevenLabs,
    voice_settings: VoiceSettings,
    model_id: str,
    output_dir: Path,
    timestamp: str
):
    """
    Iterate over the regeneration data, generating audio segments.

    data_to_regenerate tuples contain an integer representing the index
    of the item the new segment will be replacing in the original
    sequence of segments; the dict is the text, voice_id, and filepath

    Save audio segments to files. Return lookup dict/map of new filepaths,
    where fieldname is the index being replaced in the original sequence
    of audio segments.
    """
    new_filepaths_lookup_map = {}
    for ix, generation_datum_tuple in enumerate(data_to_regenerate):
        original_segment_ix, generation_datum = generation_datum_tuple
        text_string = generation_datum["text"]
        voice_id = generation_datum["voice_id"]
        previous_request_id = get_previous_request_id(data_to_regenerate, original_segment_ix)
        output_filepath = Path(output_dir / f"audio_chunk_{original_segment_ix:03d}_{timestamp}.wav")
        typer.echo(f"Generating audio chunk {ix+1} of {len(data_to_regenerate)}...")
        generate_audio_with_timeout(
            text=text_string,
            voice_id=voice_id,
            output_file=output_filepath,
            tts_client=tts_client,
            voice_settings=voice_settings,
            model_id=model_id,
            previous_request_ids=[previous_request_id]
        )
        new_filepaths_lookup_map[original_segment_ix] = str(output_filepath)
    return new_filepaths_lookup_map

def get_previous_request_id(
    data_to_regenerate: list[tuple[int, dict]],
    original_segment_ix: int
):
    """
    Get the request_id from data_to_regenerate
    items by index

    Represents the request_id of the item
    preceding the current item in the dataset
    """
    try:
        previous_request_id = (
            data_to_regenerate[original_segment_ix-1][1]['request_id'] 
            if original_segment_ix != 0 else ''
        )
    except:
        previous_request_id = ''
    
    return previous_request_id

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
    voice_settings: VoiceSettings,
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
                voice_settings=voice_settings,
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
    voice_settings: VoiceSettings,
    model_id: str = 'eleven_multilingual_v2',
    previous_request_ids: list[str] = []
):
    """
    Given a text string, generate audio using ElevenLabs
    Text to Speech API

    previous_request_ids should be from requests made
    no longer than 2 hours ago, but it appears that providing
    older or invalid IDs does not adversely impact generation
    or trigger an error (as compared to providing no IDs)

    max of 3 previous_requests_ids is accepted by the API
    """
    max_retries = 5
    delay = 0.5
    
    # max of three previous request IDs are accepted
    # https://elevenlabs.io/docs/cookbooks/text-to-speech/request-stitching
    if previous_request_ids:
        previous_request_ids = previous_request_ids[-3:]

    for attempt in range(max_retries):
        try:
            with tts_client.text_to_speech.with_raw_response.convert(
                model_id=model_id,
                text=text,
                voice_id=voice_id,
                voice_settings=voice_settings,
                previous_request_ids=previous_request_ids,
                output_format='pcm_24000' # important to use this encoding
                                        # for compatibility with write_audio_data_to_wav_file
            ) as response:
                request_id = response._response.headers.get("request-id")
                audio_data = b''.join(chunk for chunk in response.data)
                write_audio_data_to_wav_file(output_file, audio_data)
                return request_id
        except Exception as e:
            if attempt < max_retries - 1:
                # wait if we have more attempts
                jitter_wait(delay, attempt, e)

    typer.echo("Max retries exceeded. Exiting.")
    raise typer.Exit(1)   

def generate_voice_settings(user_config: dict):
    """
    Configure VoiceSettings instance from
    dict of user_config settings.

    Defaults applied in case of failure to
    find appropriate setting in user_config.
    """
    expected_fields_and_defaults = [
        ('stability', 0.5),
        ('similarity_boost', 0.75),
        ('style', 0.0),
        ('use_speaker_boost', True),
        ('speed', 1.0)
    ]
    
    final_config = {}

    for key, default_val in expected_fields_and_defaults:
        if (
            key in user_config
            and type(user_config[key]) == type(default_val)
        ):
            final_config[key] = user_config[key]
        else:
            typer.echo(
                (
                    f"User voice setting '{key}' not found or invalid. "
                    f"Using default: {default_val}"
                )
            )
            final_config[key] = default_val

    return VoiceSettings(
                stability=final_config['stability'],
                similarity_boost=final_config['similarity_boost'],
                style=final_config['style'],
                use_speaker_boost=final_config['use_speaker_boost'],
                speed=final_config['speed']
            )

def jitter_wait(delay: float, attempt: int, e: Exception = None):
    """
    Set wait time and sleep
    """
    wait = delay * (2**attempt)
    jittered_wait = wait * random.uniform(0.8, 1.2)
    error_str = "error." if not e else f"error: {e}"
    typer.echo(f"Retrying after {jittered_wait:.1f}s due to {error_str}...")
    time.sleep(jittered_wait)

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


def validate_voices(
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

    if speaker_one_voice == speaker_two_voice:
        typer.echo("Unique IDs are required for speaker_one_voice and speaker_two_voice.")
        raise typer.Exit(1)

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

def execute_audio_generation_workflow(
    transcript: str,
    output_dir: Path,
    timestamp: str,
    speaker_one_voice: str,
    speaker_two_voice: str,
    tts_client: ElevenLabs,
    voice_settings: VoiceSettings,
    tts_model: str,
    backup_filepath: Path
):
    """
    Workflow for generating the first pass 
    of audio segments.

    Return a list of filepaths to the audio segments
    """
    # chunk transcript
    typer.echo("Chunking transcript. This may take a few minutes...")
    transcript_chunks = create_speaker_text_chunks(transcript)
    typer.echo(f"Transcript split into {len(transcript_chunks)} chunks.")

    # create generation plan
    generation_data = create_generation_data(
        transcript_chunks,
        output_dir,
        timestamp,
        speaker_one_voice,
        speaker_two_voice
    )
    audio_chunk_filepaths = [x['filepath'] for x in generation_data]
        
    # generate audio
    typer.echo(
        "Generating audio from transcript chunks. "
        "This could take a while..."
    )
    generate_audio_segments(
        generation_data, 
        tts_client, 
        voice_settings, 
        tts_model, 
        backup_filepath
    )

    return audio_chunk_filepaths

def execute_audio_regeneration_workflow(
    backup_file_for_regen: Path,
    tts_client: ElevenLabs,
    voice_settings: VoiceSettings,
    tts_model: str,
    timestamp: str
) -> tuple[Path, list[Path]]:
    """
    Workflow for regenerating the selected 
    audio segments.

    Return (output_dir, audio_chunk_filepaths) 
        audio_chunk_filepaths: list of filepaths to the newly generated audio segments
        interspersed, where appropriate, among the original audio segments.
    """
    # get previous generation data
    all_generation_data = read_backup_from_json_file(backup_file_for_regen, tts_client)

    # set output directory
    output_dir = Path(all_generation_data[1]['filepath']).parent

    # user selects segments to regen
    ix_of_items_to_regen = generate_menu(
        [Path(x.get('filepath')).name for x in all_generation_data],
        'Select audio segments to regenerate',
        multi_select=True
    )

    # exit in case of, e.g., Ctrl + C
    if ix_of_items_to_regen is None:
        typer.echo('Exiting.')
        raise typer.Exit(0)

    # select original data of items to regenerate, by index
    data_to_regenerate = [(i, all_generation_data[i]) for i in ix_of_items_to_regen]
    
    # regen segments
    new_filepaths_lookup_map = regenerate_audio_segments(
        data_to_regenerate, 
        tts_client, 
        voice_settings,
        tts_model, 
        output_dir, 
        timestamp
    )
    
    # collect updated and original segment filepaths
    audio_chunk_filepaths = [
        new_filepaths_lookup_map.get(i, x['filepath']) 
        for i, x in enumerate(all_generation_data)
    ]

    return output_dir, audio_chunk_filepaths



if __name__ == "__main__":
    app()
