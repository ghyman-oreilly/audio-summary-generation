# Audio Summary Generator

A set of tools for generating a podcast-style audio summary from book PDF.

## Requirements

* Python 3.11+
* [Gemini API or Google API key](https://ai.google.dev/gemini-api/docs/api-key)
* ElevenLabs API key

## Setup

1. Clone the repository or download the source files:

	```bash
	git clone git@github.com:ghyman-oreilly/audio-summary-generation.git
	
	cd audio-summary-generation
	```

2. Install required dependencies:

	```bash
	pip install -r requirements.txt
	```

## Usage

To use the script, run the following command, providing the path to the PDF you want to base your audio summary on:

```bash
python main.py <path/to/your.pdf>
```

The overall flow is as follows: 

1. The script generates a text summary of the PDF.
2. Based on your response to a prompt, the script stops or proceeds to generate a podcast-style transcript from the  summary.
3. Based on your response to a prompt, the script stops or proceeds to generate a podcast-style audio summary from the transcript.

The prompts at steps 2 and 3 are designed to give you an opportunity to edit the summary and transcript, if/as needed, before rerunning the script.

The following options can be used to skip steps, when appropriate (e.g., when inputting a previously generated and now edited transcript), and to otherwise customize the script experience and output:

- `--output_dir`: Set the output directory for text (summary and transcript) and audio. Defaults to current working directory.
- `--text_summary_file`: Input a previously generated text summary file to skip generation of a new one. With this option, the PDF path is not required.
- `--transcript_file`: Input a previously generated text transcript file to skip generation of a new one. With this option, the PDF path is not required.
- `--speaker-one-voice`: Choose the voice to use for the first speaker. Defaults to random. See https://ai.google.dev/gemini-api/docs/speech-generation for options.
- `--speaker-two-voice`: Choose the voice to use for the second speaker. Defaults to random.
- `--speaker-one-prefix`: Choose the prefix used for the first speaker in your transcript. Defaults to 'Speaker 1'.
- `--speaker-two-prefix`: Choose the prefix used for the second speaker in your transcript. Defaults to 'Speaker 2'.
