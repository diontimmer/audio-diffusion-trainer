import PySimpleGUI as sg
from threading import Thread

sg.theme('DarkGrey7')
output_frame = [
    [sg.Multiline(disabled=True, size=(200, 20), autoscroll=True, auto_refresh=True, key='-LOG-', reroute_stdout=True, reroute_stderr=False)]
]
input_frame = [
    [sg.Multiline(size=(200, 10), key='-PROMPTS-')],
    [sg.Button('Generate')],
]


layout = [
    [sg.Frame('Output', output_frame)],
    [sg.Frame('Input', input_frame)],
]

window = sg.Window('Prompt Generator', layout)

from sd_gui_helpers import *

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Generate':
        Thread(target=generate_audio, args=(values,), daemon=True).start()

window.close()