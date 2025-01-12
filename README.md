<div align=center><h1>OpenAI-Whisper for *Macbook Pro M3 Max | Windows</h1></div>
*Testet on Macbook Pro M3 Max with 36GB*

## This program supports the following video formats: .mp4, .avi, .mkv, .mov

# Install on Mac
You need a package manager, my recommendation is Homebrew.
Website:

```sh
https://brew.sh
```

To do this, open your terminal and execute the following command:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
```sh
brew update
```
```sh
brew install ffmpeg
```

For main.py please install 2 models

```sh
ollama pull llama3.2-vision
```
```sh
ollama pull llava:13b
```

Install Git
<https://git-scm.com/downloads>

Install Python
<https://www.python.org/downloads/>

<h2>Clone Repository</h2>

```sh
git clone https://github.com/MarkusR1805/Image-to-Prompt.git
```

<h2>OPTIONAL!! Create python venv</h2>

```sh
python -m venv image-to-prompt
cd image-to-prompt
source bin/activate
```

<h1>Attention, very important!</h1>
If the program does not start or an error message appears, be sure to execute the requirements.txt.

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

<h2>Programm start</h2>

```sh
python main.py
```

![Promptgenerator](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/26f2122f-6738-45e1-bcf9-0e62f281622c/original=true,quality=90/36686347.jpeg)
