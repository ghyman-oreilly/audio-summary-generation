from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import os
import time
import typer
from typing import Optional

from prompts import TEXT_SUMMARY_PROMPT, TRANSCRIPT_SYS_INSTRUCTIONS


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
        )
):
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # check for API key
    if not api_key:
        typer.echo("GOOGLE_API_KEY not found in environment. Exiting...")
        typer.exit(1)

    timestamp = int(time.time())
    
    # check/config output directory
    if output_dir:
        if not dir_is_valid(output_dir):
            typer.echo("Exiting...")
            typer.Exit(1)
    else:
        output_dir = Path.cwd()

    # generate text summary
    if path_to_pdf and not text_summary_file:	
        if not file_is_valid(path_to_pdf, '.pdf', 20):
            typer.echo("Exiting...")
            typer.Exit(1)

        typer.echo("Generating text summary from PDF. This may take a few minutes...")
        text_summary = infer_with_pdf_document_understanding(path_to_pdf, TEXT_SUMMARY_PROMPT, api_key)

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

    # handle existing/inputted text summary
    if text_summary_file:
        if not file_is_valid(text_summary_file, '.txt'):
            typer.echo("Exiting...")
            typer.Exit(1)
        text_summary = read_text_from_file(text_summary_file)
    
    # generate transcript
    typer.echo("Generating transcript from text summary. This may take a few minutes...")
    transcript = generate_text(text_summary, TRANSCRIPT_SYS_INSTRUCTIONS, api_key)
    transcript_output_path = Path(output_dir / f'transcript_{timestamp}.txt')
    write_text_to_file(transcript, transcript_output_path)
    typer.echo(f"Transcript written to {transcript_output_path}")

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
        api_key: Optional[str] = None
) -> str:
    """
    Given a PDF file as corpus, generate a response to a user prompt
    """
    if api_key:
        client = genai.Client(api_key=api_key)
    else: 
        client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
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
        sys_instrux: Optional[str] = None,
        api_key: Optional[str] = None
) -> str:
    """
    Generate text, given a user prompt
    and optional system instructions

    user_prompt can be a natural language prompt
    or a text that the system instructions indicate
    the model should transform
    """
    if api_key:
        client = genai.Client(api_key=api_key)
    else: 
        client = genai.Client()

    if sys_instrux:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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


if __name__ == "__main__":
    typer.run(main)