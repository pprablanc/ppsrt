# Python-pitch-shifting-RT
A python algorithm to change the pitch of the voice in real time. It is also possible to control the pitch shifting rate in real time in the command line interface.

## Requirements

- pyaudio
- readchar


## Installation

``` shell
git clone https://github.com/pprablanc/ppsrt.git
cd ppsrt
pip install -e .
```

## Usage
In the terminal:
``` shell
cd ppsrt
./ppsrt.py
```

In the terminal interface, you can increase/decrease the pitch shift of the voice with '+'/'-' keys respectively.
To leave the program, hit space twice.


## Warnings
If you use this program within Spyder IDE, you might have problems with readchar. It is recommended to use it in a standard terminal.
