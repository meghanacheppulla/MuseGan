"""MuseGAN music generator - desktop GUI (Tkinter).

Run:   python gui.py
Tab 1 trains the GAN (live loss chart), tab 2 generates songs, shows the piano roll
and plays the MIDI file. Needs musegan.py in the same folder (it is part of this project).
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "musegan_generator.pth")
OUT_DIR = os.path.join(HERE, "output")
SONG_CHOICES = {"64 songs (quick test)": 64, "256 songs": 256, "512 songs (best)": 512}
TRACK_NAMES = ["Drums", "Bass", "Piano", "Melody"]
TRACK_COLORS = [(214, 96, 77), (67, 147, 195), (77, 172, 130), (244, 165, 61)]


# --------------------------------------------------------------------------
# Engine (heavy libraries are imported only when needed so the window opens fast)
# --------------------------------------------------------------------------
def fit(epochs, n_songs, on_log, on_progress, on_epoch, should_stop):
    import torch
    import musegan as mg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_log("Device: %s" % device)
    on_log("Creating %d synthetic training songs ..." % n_songs)
    data = torch.from_numpy(mg.synthetic_dataset(n_samples=n_songs))
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, drop_last=True)
    gen, critic = mg.Generator().to(device), mg.Critic().to(device)
    g_opt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.9))
    c_opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9))
    total = epochs * len(loader)
    done = 0
    t0 = time.time()
    on_log("Training %d epoch(s) ..." % epochs)
    for ep in range(1, epochs + 1):
        c_sum = g_sum = 0.0
        g_n = 0
        for i, real in enumerate(loader):
            if should_stop():
                on_log("Stopped. The generator saved after the last finished epoch is kept.")
                return
            real = real.to(device)
            with torch.no_grad():
                fake = gen(gen.sample_latent(real.size(0), device))
            c_loss = (critic(fake).mean() - critic(real).mean()
                      + 10.0 * mg.gradient_penalty(critic, real, fake, device))
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()
            c_sum += c_loss.item()
            if i % 2 == 0:
                g_loss = -critic(gen(gen.sample_latent(real.size(0), device))).mean()
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_sum += g_loss.item()
                g_n += 1
            done += 1
            elapsed = time.time() - t0
            on_progress(done, total, elapsed / done * (total - done))
        torch.save(gen.state_dict(), CKPT)
        on_epoch(ep, c_sum / len(loader), g_sum / max(1, g_n))
    on_log("Finished. Generator saved as musegan_generator.pth")


def generate(n):
    import torch
    import musegan as mg

    if not os.path.isfile(CKPT):
        raise FileNotFoundError("No trained generator yet. Train it first in tab 1.")
    device = torch.device("cpu")
    gen = mg.Generator().to(device)
    gen.load_state_dict(torch.load(CKPT, map_location=device))
    gen.eval()
    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        rolls = gen(gen.sample_latent(n, device)).cpu().numpy()
    stamp = time.strftime("%H%M%S")
    songs = []
    for i, roll in enumerate(rolls, 1):
        path = os.path.join(OUT_DIR, "song_%s_%d.mid" % (stamp, i))
        mg.to_midi(roll, path, 0.5)
        songs.append((roll, path))
    return songs


def render_roll(roll, threshold=0.5, cell=10):
    """Draw a (tracks, bars, steps, pitches) array as a coloured piano roll picture."""
    import numpy as np

    n_tracks, bars, steps_per_bar, pitches = roll.shape
    steps = bars * steps_per_bar
    label_w, strip_h, gap = 66, 96, 8
    width = label_w + steps * cell + 8
    height = n_tracks * (strip_h + gap) + gap
    img = Image.new("RGB", (width, height), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for t in range(n_tracks):
        y0 = gap + t * (strip_h + gap)
        d.rectangle([label_w, y0, label_w + steps * cell, y0 + strip_h], fill=(255, 255, 255),
                    outline=(190, 190, 190))
        d.text((6, y0 + strip_h // 2 - 5), TRACK_NAMES[t % len(TRACK_NAMES)], fill=(60, 60, 60))
        for b in range(1, bars):
            x = label_w + b * steps_per_bar * cell
            d.line([x, y0, x, y0 + strip_h], fill=(215, 215, 215))
        notes = roll[t].reshape(steps, pitches) > threshold
        color = TRACK_COLORS[t % len(TRACK_COLORS)]
        if t == 0:  # drums: three rows (kick, snare, hi-hat)
            row_h = strip_h / 3.0
            for s, p in zip(*np.nonzero(notes[:, :3])):
                y = y0 + strip_h - (p + 1) * row_h
                d.rectangle([label_w + s * cell + 1, y + 2, label_w + (s + 1) * cell - 1, y + row_h - 2], fill=color)
        else:
            row_h = strip_h / float(pitches)
            for s, p in zip(*np.nonzero(notes)):
                y = y0 + strip_h - (p + 1) * row_h
                d.rectangle([label_w + s * cell, y, label_w + (s + 1) * cell - 1, y + max(1.5, row_h - 0.3)], fill=color)
    return img


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: opens in the default player
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("MuseGAN Music Generator")
        root.minsize(780, 700)
        self.q = queue.Queue()
        self.stop_flag = False
        self.thread = None
        self.total_epochs = 5
        self.c_losses, self.g_losses = [], []
        self.songs = []
        self.photo = None
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.nb = nb
        self.train_tab = ttk.Frame(nb)
        self.gen_tab = ttk.Frame(nb)
        nb.add(self.train_tab, text="  1. Train the GAN  ")
        nb.add(self.gen_tab, text="  2. Generate music  ")
        self.build_train()
        self.build_gen()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll()

    # ---- tab 1: training --------------------------------------------------
    def build_train(self):
        f = self.train_tab
        ttk.Label(f, text="Train MuseGAN on synthetic multi-track music",
                  font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=6, pady=(12, 2))
        ttk.Label(f, text="The generator is saved after every epoch, so you can generate music at any time.",
                  foreground="#555").grid(row=1, column=0, columnspan=6, pady=(0, 10))
        ttk.Label(f, text="Epochs").grid(row=2, column=0, sticky="e", padx=4)
        self.epochs_var = tk.IntVar(value=30)
        ttk.Spinbox(f, from_=1, to=300, width=5, textvariable=self.epochs_var).grid(row=2, column=1, sticky="w")
        ttk.Label(f, text="Training songs").grid(row=2, column=2, sticky="e", padx=4)
        self.songs_var = tk.StringVar(value="256 songs")
        ttk.Combobox(f, textvariable=self.songs_var, values=list(SONG_CHOICES), state="readonly",
                     width=22).grid(row=2, column=3, sticky="w")
        self.start_btn = ttk.Button(f, text="Start training", command=self.start_train)
        self.start_btn.grid(row=2, column=4, padx=8)
        self.stop_btn = ttk.Button(f, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.grid(row=2, column=5, padx=4)

        self.progress = ttk.Progressbar(f, length=640, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=6, pady=(14, 2))
        self.status = ttk.Label(f, text="Press 'Start training'.")
        self.status.grid(row=4, column=0, columnspan=6)

        self.chart = tk.Canvas(f, width=640, height=230, bg="white", highlightthickness=1,
                               highlightbackground="#bbb")
        self.chart.grid(row=5, column=0, columnspan=6, pady=10)
        self.draw_chart()

        box = ttk.Frame(f)
        box.grid(row=6, column=0, columnspan=6, padx=10, sticky="ew")
        self.log = tk.Text(box, height=8, width=86, state="disabled", wrap="word")
        sb = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def draw_chart(self):
        c = self.chart
        c.delete("all")
        left, right, top, bottom = 56, 620, 28, 194
        c.create_text(320, 10, text="Training losses (lower is not always better for a GAN)",
                      font=("Segoe UI", 10, "bold"))
        c.create_rectangle(left, top, right, bottom, outline="#ddd")
        series = (("Critic", self.c_losses, "#2b6cb0"), ("Generator", self.g_losses, "#d9822b"))
        values = self.c_losses + self.g_losses
        if values:
            lo, hi = min(values), max(values)
            if hi - lo < 1e-6:
                lo, hi = lo - 1, hi + 1
        else:
            lo, hi = -1.0, 1.0
        for k in range(5):
            val = lo + (hi - lo) * k / 4
            y = bottom - (bottom - top) * k / 4
            c.create_line(left, y, right, y, fill="#eee")
            c.create_text(left - 6, y, text="%.1f" % val, anchor="e", fill="#555")
        n = max(2, self.total_epochs)
        step = (right - left) / n
        for name, vals, color in series:
            pts = [(left + step * (i + 0.5), bottom - (bottom - top) * (v - lo) / (hi - lo))
                   for i, v in enumerate(vals)]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                c.create_line(x1, y1, x2, y2, fill=color, width=2)
            for x, y in pts:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
        c.create_rectangle(left + 8, bottom + 12, left + 20, bottom + 22, fill="#2b6cb0", outline="")
        c.create_text(left + 26, bottom + 17, text="Critic loss", anchor="w")
        c.create_rectangle(left + 130, bottom + 12, left + 142, bottom + 22, fill="#d9822b", outline="")
        c.create_text(left + 148, bottom + 17, text="Generator loss", anchor="w")
        c.create_text(right - 4, bottom + 17, text="Epoch 1 to %d" % self.total_epochs, anchor="e", fill="#555")

    def add_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_train(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            epochs = int(self.epochs_var.get())
            if not 1 <= epochs <= 300:
                raise ValueError
        except Exception:
            messagebox.showinfo("Epochs", "Please enter a number of epochs between 1 and 300.")
            return
        n_songs = SONG_CHOICES[self.songs_var.get()]
        self.total_epochs = epochs
        self.c_losses, self.g_losses = [], []
        self.draw_chart()
        self.stop_flag = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.status.configure(text="Starting ...")
        self.thread = threading.Thread(target=self.train_worker, args=(epochs, n_songs), daemon=True)
        self.thread.start()

    def train_worker(self, epochs, n_songs):
        try:
            fit(epochs, n_songs,
                on_log=lambda m: self.q.put(("log", m)),
                on_progress=lambda d, t, eta: self.q.put(("progress", d, t, eta)),
                on_epoch=lambda ep, c, g: self.q.put(("epoch", ep, c, g)),
                should_stop=lambda: self.stop_flag)
            self.q.put(("train_done",))
        except Exception as e:
            self.q.put(("error", "%s: %s" % (type(e).__name__, e)))

    def stop(self):
        self.stop_flag = True
        self.status.configure(text="Stopping after the current batch ...")

    # ---- tab 2: generate --------------------------------------------------
    def build_gen(self):
        f = self.gen_tab
        ttk.Label(f, text="Generate new music", font=("Segoe UI", 14, "bold")
                  ).grid(row=0, column=0, columnspan=6, pady=(12, 2))
        ttk.Label(f, text="Each song has 4 bars and 4 instruments: drums, bass, piano and melody.",
                  foreground="#555").grid(row=1, column=0, columnspan=6, pady=(0, 8))
        ttk.Label(f, text="How many songs").grid(row=2, column=0, sticky="e", padx=4)
        self.num_var = tk.IntVar(value=3)
        ttk.Spinbox(f, from_=1, to=10, width=4, textvariable=self.num_var).grid(row=2, column=1, sticky="w")
        self.gen_btn = ttk.Button(f, text="Generate", command=self.start_generate)
        self.gen_btn.grid(row=2, column=2, padx=10)
        ttk.Label(f, text="Song").grid(row=2, column=3, sticky="e", padx=4)
        self.song_var = tk.StringVar()
        self.song_box = ttk.Combobox(f, textvariable=self.song_var, values=[], state="disabled", width=12)
        self.song_box.grid(row=2, column=4, sticky="w")
        self.song_box.bind("<<ComboboxSelected>>", lambda e: self.show_song())

        self.roll_label = ttk.Label(f)
        self.roll_label.grid(row=3, column=0, columnspan=6, pady=12)
        self.show_roll(None)
        row = ttk.Frame(f)
        row.grid(row=4, column=0, columnspan=6, pady=4)
        self.play_btn = ttk.Button(row, text="Play in my music player", command=self.play, state="disabled")
        self.play_btn.pack(side="left", padx=6)
        self.folder_btn = ttk.Button(row, text="Open output folder", command=self.open_folder, state="disabled")
        self.folder_btn.pack(side="left", padx=6)
        self.save_btn = ttk.Button(row, text="Save a copy ...", command=self.save_copy, state="disabled")
        self.save_btn.pack(side="left", padx=6)
        self.gen_status = ttk.Label(f, text="Train the model first (tab 1), then press Generate.")
        self.gen_status.grid(row=5, column=0, columnspan=6, pady=8)

    def show_roll(self, pil):
        if pil is None:
            pil = Image.new("RGB", (66 + 640 + 8, 4 * 104 + 8), "#ececec")
            ImageDraw.Draw(pil).text((290, 200), "The piano roll will appear here", fill=(120, 120, 120))
        self.photo = ImageTk.PhotoImage(pil)
        self.roll_label.configure(image=self.photo)

    def start_generate(self):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Please wait", "Training is still running.")
            return
        try:
            n = int(self.num_var.get())
            if not 1 <= n <= 10:
                raise ValueError
        except Exception:
            messagebox.showinfo("Songs", "Please enter a number of songs between 1 and 10.")
            return
        self.gen_btn.configure(state="disabled")
        self.gen_status.configure(text="Generating ...")
        threading.Thread(target=self.gen_worker, args=(n,), daemon=True).start()

    def gen_worker(self, n):
        try:
            songs = generate(n)
            images = [render_roll(roll) for roll, _ in songs]
            self.q.put(("songs", [p for _, p in songs], images))
        except Exception as e:
            self.q.put(("gen_error", str(e) if isinstance(e, FileNotFoundError)
                        else "%s: %s" % (type(e).__name__, e)))

    def show_song(self):
        i = self.song_box.current()
        if 0 <= i < len(self.songs):
            path, img = self.songs[i]
            self.show_roll(img)
            self.gen_status.configure(text="Saved as " + path)

    def current_path(self):
        i = self.song_box.current()
        return self.songs[i][0] if 0 <= i < len(self.songs) else None

    def play(self):
        path = self.current_path()
        if not path:
            return
        try:
            open_file(path)
            self.gen_status.configure(text="Opened in your default player. (MIDI files need a MIDI-capable player.)")
        except Exception as e:
            messagebox.showerror("Cannot play", "Could not open the file (%s).\nOpen it from the output folder instead." % e)

    def open_folder(self):
        try:
            open_file(OUT_DIR)
        except Exception as e:
            messagebox.showerror("Cannot open folder", str(e))

    def save_copy(self):
        path = self.current_path()
        if not path:
            return
        dest = filedialog.asksaveasfilename(defaultextension=".mid", initialfile=os.path.basename(path),
                                            filetypes=[("MIDI file", "*.mid")])
        if dest:
            with open(path, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
            self.gen_status.configure(text="Saved " + dest)

    # ---- shared -----------------------------------------------------------
    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.add_log(msg[1])
                elif kind == "progress":
                    _, done, total, eta = msg
                    self.progress.configure(maximum=total, value=done)
                    self.status.configure(text="%d%% done   -   about %d min %02d s left" %
                                          (100 * done // total, eta // 60, eta % 60))
                elif kind == "epoch":
                    _, ep, c, g = msg
                    self.c_losses.append(c)
                    self.g_losses.append(g)
                    self.draw_chart()
                    self.add_log("Epoch %d/%d   critic %.3f   generator %.3f" % (ep, self.total_epochs, c, g))
                elif kind == "train_done":
                    self.finish_train("Training finished. Open tab 2 to generate music.")
                elif kind == "songs":
                    paths, images = msg[1], msg[2]
                    self.songs = list(zip(paths, images))
                    names = ["Song %d" % (i + 1) for i in range(len(paths))]
                    self.song_box.configure(values=names, state="readonly")
                    self.song_box.current(0)
                    self.show_song()
                    for b in (self.play_btn, self.folder_btn, self.save_btn):
                        b.configure(state="normal")
                    self.gen_btn.configure(state="normal")
                elif kind == "gen_error":
                    self.gen_btn.configure(state="normal")
                    self.gen_status.configure(text="Could not generate.")
                    messagebox.showinfo("Cannot generate", msg[1])
                elif kind == "error":
                    self.finish_train("Something went wrong.")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def finish_train(self, text):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.configure(text=text)

    def close(self):
        self.stop_flag = True
        self.root.destroy()


def main():
    try:  # sharper text on Windows high-DPI screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
