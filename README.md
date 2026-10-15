# Project 5 - MuseGAN Music Generator (with GUI)

A simplified **MuseGAN** in PyTorch that generates 4-bar, 4-instrument music (drums, bass, piano,
melody) as MIDI files. It is a Wasserstein GAN with gradient penalty. No dataset download is needed:
the program creates its own multi-track training songs.

## Step by step (Windows)

1. Install Python 3.9 or newer from python.org. Tick **"Add Python to PATH"**.
2. Extract the zip. Open the extracted folder until you see `gui.py`.
3. Click the address bar of File Explorer, type `powershell` and press Enter.
4. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   (If PowerShell blocks it, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.)
5. Install the packages:
   ```
   pip install -r requirements.txt
   ```
6. Start the program:
   ```
   python gui.py
   ```

## Using the window

**Tab 1 - Train the GAN**
1. Leave the defaults (30 epochs, 256 songs) and click **Start training**.
   For a quick test choose 3 epochs and "64 songs".
2. Watch the progress bar and the loss chart. The generator is saved after every epoch.

**Tab 2 - Generate music**
1. Choose how many songs and click **Generate**.
2. The piano roll appears (one coloured strip per instrument). Pick another song in the **Song** box.
3. **Play in my music player** opens the MIDI file in your default player (Windows Media Player and
   VLC play .mid files). **Open output folder** shows all songs; **Save a copy ...** exports one.

More epochs give cleaner, more musical results.

## Command line (optional)

```
python musegan.py train --epochs 30
python musegan.py generate --num 3
```

## Troubleshooting

* "No trained generator yet" : train first in tab 1.
* `No module named torch` / `pretty_midi` : the venv is not active, or step 5 was skipped.
* The MIDI file will not play : install VLC (videolan.org) and set it as the default for .mid files.
* `No module named tkinter` : reinstall Python and keep **tcl/tk and IDLE** ticked.
