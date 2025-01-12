<div align=center><h1>OpenAI-Whisper for *Macbook Pro M3 Max | Windows</h1></div>
*Testet on Macbook Pro M3 Max with 36GB <p>

![openai-whisper](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/5f219a87-05c9-4510-bd4c-eb5856628332/original=true,quality=90/42965033.jpeg)

## This program supports the following video formats: .mp4, .avi, .mkv, .mov

# Install on Mac
## You need a package manager, my recommendation is Homebrew.
## Website:
```sh
https://brew.sh
```

## Install Homebrew
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

## You need Git on your System

Install Git
<https://git-scm.com/downloads>

Install Python
<https://www.python.org/downloads/>

<h2>Clone Repository</h2>

```sh
git clone https://github.com/MarkusR1805/whisper-transcribe.git
```

## These Python libraries are installed by the requirements.txt

certifi==2024.12.14<br>
charset-normalizer==3.4.1<br>
ffmpeg-python==0.2.0<br>
filelock==3.16.1<br>
fsspec==2024.12.0<br>
future==1.0.0<br>
idna==3.10<br>
Jinja2==3.1.5<br>
llvmlite==0.43.0<br>
MarkupSafe==3.0.2<br>
more-itertools==10.5.0<br>
mpmath==1.3.0<br>
networkx==3.4.2<br>
numba==0.60.0<br>
numpy==2.0.2<br>
openai-whisper==20240930<br>
psutil==6.1.1<br>
PyQt6==6.8.0<br>
PyQt6-Qt6==6.8.1<br>
PyQt6_sip==13.9.1<br>
regex==2024.11.6<br>
requests==2.32.3<br>
setuptools==75.8.0<br>
six==1.17.0<br>
sympy==1.13.1<br>
tiktoken==0.8.0<br>
torch==2.5.1<br>
tqdm==4.67.1<br>
typing_extensions==4.12.2<br>
urllib3==2.3.0<br>

<h2>OPTIONAL!! Create python venv</h2>

```sh
python -m venv whisper-transcribe
cd whisper-transcribe
source bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<h2>Programm start</h2>

```sh
python main.py
```

![openai-whisper](https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/ccc78288-c4b1-4660-af9b-6856c860dc67/original=true,quality=90/48383996.jpeg)

<video width="320" height="240" controls>
  <source src="https://youtu.be/FxCxbUwAnZQ" type="video/mp4">
</video>
