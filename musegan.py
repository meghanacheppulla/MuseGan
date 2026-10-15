"""Simplified MuseGAN: multi-track symbolic music generation with a GAN (PyTorch).

Tensor layout:  (batch, tracks, bars, steps_per_bar, pitches)
Tracks: drums, bass, piano (chords), melody.

Commands:
    python musegan.py train    --epochs 30            # trains on synthetic data and saves a checkpoint
    python musegan.py generate --num 3                # writes generated_*.mid files
    python musegan.py train --data pianorolls.npy     # train on your own data (see README)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

TRACKS = ["Drums", "Bass", "Piano", "Melody"]
N_TRACKS = len(TRACKS)
N_BARS = 4
STEPS = 16          # sixteenth notes per bar
PITCHES = 48        # pitch range (MIDI 36 .. 83)
LOW_PITCH = 36
Z_DIM = 32
TEMPO = 110


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def synthetic_dataset(n_samples, seed=0):
    """Make simple but structured multi-track songs: chord progression, bass line,
    drum pattern and a melody drawn from the chord notes."""
    rng = np.random.RandomState(seed)
    data = np.zeros((n_samples, N_TRACKS, N_BARS, STEPS, PITCHES), dtype=np.float32)
    scale = [0, 2, 4, 5, 7, 9, 11]                       # C major
    progressions = [[0, 3, 4, 3], [0, 5, 3, 4], [0, 4, 5, 3], [5, 3, 0, 4]]
    for n in range(n_samples):
        prog = progressions[rng.randint(len(progressions))]
        root_shift = rng.randint(0, 5)
        kick = rng.rand(STEPS) < 0.0
        kick[[0, 8]] = True
        for bar in range(N_BARS):
            degree = prog[bar]
            root = 12 + scale[degree % 7] + root_shift   # offset inside PITCHES range
            third = root + scale[(degree + 2) % 7] - scale[degree % 7] + (12 if (degree + 2) % 7 < degree % 7 else 0)
            fifth = root + scale[(degree + 4) % 7] - scale[degree % 7] + (12 if (degree + 4) % 7 < degree % 7 else 0)
            chord = [p for p in (root, third, fifth) if 0 <= p < PITCHES]
            for s in range(STEPS):
                # drums: kick on 0/8, snare on 4/12, hi-hat every 2 steps
                if kick[s]:
                    data[n, 0, bar, s, 0] = 1
                if s in (4, 12):
                    data[n, 0, bar, s, 1] = 1
                if s % 2 == 0:
                    data[n, 0, bar, s, 2] = 1
                # bass: root every 4 steps, one octave lower
                if s % 4 == 0 and root - 12 >= 0:
                    data[n, 1, bar, s, root - 12] = 1
                # piano: chord stabs on steps 0 and 8
                if s in (0, 8):
                    for p in chord:
                        data[n, 2, bar, s, p] = 1
                # melody: notes from the chord / scale on 8th notes
                if s % 2 == 0 and rng.rand() < 0.8:
                    p = chord[rng.randint(len(chord))] + 12
                    if p < PITCHES:
                        data[n, 3, bar, s, p] = 1
    return data


def load_data(path, n_synth):
    if path:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 5 or arr.shape[1:] != (N_TRACKS, N_BARS, STEPS, PITCHES):
            raise SystemExit("Data must have shape (N, %d, %d, %d, %d), got %s"
                             % (N_TRACKS, N_BARS, STEPS, PITCHES, arr.shape))
        return (arr > 0).astype(np.float32)
    return synthetic_dataset(n_samples=n_synth)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class BarGenerator(nn.Module):
    """Turns a latent vector into one bar (STEPS x PITCHES) for one track."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 128 * 2 * 3), nn.BatchNorm1d(128 * 2 * 3), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),   # 2x3  -> 4x6
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),    # 4x6  -> 8x12
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),    # 8x12 -> 16x24
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=(1, 2), stride=(1, 2)),     # 16x24 -> 16x48
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 2, 3)
        return torch.sigmoid(self.net(x)).squeeze(1)       # (B, STEPS, PITCHES)


class Generator(nn.Module):
    """Latent = chords (shared across tracks, evolves over bars) + style (fixed) +
    melody (per track, evolves over bars) + groove (per track, fixed)."""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.chords_net = nn.Sequential(nn.Linear(z_dim, z_dim * 2), nn.ReLU(True),
                                        nn.Linear(z_dim * 2, N_BARS * z_dim))
        self.melody_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(z_dim, z_dim * 2), nn.ReLU(True),
                          nn.Linear(z_dim * 2, N_BARS * z_dim)) for _ in range(N_TRACKS)])
        self.bar_gens = nn.ModuleList([BarGenerator(4 * z_dim) for _ in range(N_TRACKS)])

    def sample_latent(self, batch, device):
        return (torch.randn(batch, self.z_dim, device=device),
                torch.randn(batch, self.z_dim, device=device),
                torch.randn(batch, N_TRACKS, self.z_dim, device=device),
                torch.randn(batch, N_TRACKS, self.z_dim, device=device))

    def forward(self, latent):
        chords, style, melody, groove = latent
        B = chords.size(0)
        c = self.chords_net(chords).view(B, N_BARS, self.z_dim)
        tracks = []
        for t in range(N_TRACKS):
            m = self.melody_nets[t](melody[:, t]).view(B, N_BARS, self.z_dim)
            bars = []
            for b in range(N_BARS):
                z = torch.cat([c[:, b], style, m[:, b], groove[:, t]], dim=1)
                bars.append(self.bar_gens[t](z))
            tracks.append(torch.stack(bars, dim=1))           # (B, BARS, STEPS, PITCHES)
        return torch.stack(tracks, dim=1)                     # (B, TRACKS, BARS, STEPS, PITCHES)


class Critic(nn.Module):
    """Wasserstein critic with a 3D-conv over (bars*tracks, steps, pitches)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(N_TRACKS, 32, kernel_size=(1, 4, 6), stride=(1, 2, 3), padding=(0, 1, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(32, 64, kernel_size=(2, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=(2, 4, 3), stride=(1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.2, True))
        with torch.no_grad():
            n = self.net(torch.zeros(1, N_TRACKS, N_BARS, STEPS, PITCHES)).numel()
        self.fc = nn.Linear(n, 1)

    def forward(self, x):                                      # x: (B, TRACKS, BARS, STEPS, PITCHES)
        return self.fc(self.net(x).flatten(1))


def gradient_penalty(critic, real, fake, device):
    eps = torch.rand(real.size(0), 1, 1, 1, 1, device=device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out = critic(mix)
    grad = torch.autograd.grad(out.sum(), mix, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


# --------------------------------------------------------------------------
# MIDI export
# --------------------------------------------------------------------------
def to_midi(roll, path, threshold=0.5):
    """roll: array (TRACKS, BARS, STEPS, PITCHES) of 0..1  ->  .mid file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)
    step_len = 60.0 / TEMPO / 4.0
    programs = {1: 33, 2: 0, 3: 73}            # bass, piano, flute
    drum_notes = {0: 36, 1: 38, 2: 42}         # kick, snare, hi-hat
    for t in range(N_TRACKS):
        inst = pretty_midi.Instrument(program=programs.get(t, 0), is_drum=(t == 0), name=TRACKS[t])
        seq = (roll[t] > threshold).reshape(N_BARS * STEPS, PITCHES)
        if t == 0:
            for s in range(seq.shape[0]):
                for p, midi_note in drum_notes.items():
                    if seq[s, p]:
                        inst.notes.append(pretty_midi.Note(90, midi_note, s * step_len, (s + 1) * step_len))
        else:
            for p in range(PITCHES):
                s = 0
                while s < seq.shape[0]:
                    if seq[s, p]:
                        e = s
                        while e + 1 < seq.shape[0] and seq[e + 1, p]:
                            e += 1
                        inst.notes.append(pretty_midi.Note(80, LOW_PITCH + p, s * step_len, (e + 1) * step_len))
                        s = e + 1
                    else:
                        s += 1
        pm.instruments.append(inst)
    pm.write(path)


# --------------------------------------------------------------------------
# Train / generate
# --------------------------------------------------------------------------
def train(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    data = torch.from_numpy(load_data(a.data, a.samples))
    print("Training data:", tuple(data.shape))
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch_size, shuffle=True, drop_last=True)
    gen, critic = Generator().to(device), Critic().to(device)
    g_opt = torch.optim.Adam(gen.parameters(), lr=a.lr, betas=(0.5, 0.9))
    c_opt = torch.optim.Adam(critic.parameters(), lr=a.lr, betas=(0.5, 0.9))

    for epoch in range(1, a.epochs + 1):
        c_sum = g_sum = 0.0
        for i, real in enumerate(loader):
            real = real.to(device)
            # --- critic step
            with torch.no_grad():
                fake = gen(gen.sample_latent(real.size(0), device))
            c_loss = (critic(fake).mean() - critic(real).mean()
                      + 10.0 * gradient_penalty(critic, real, fake, device))
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()
            c_sum += c_loss.item()
            # --- generator step every 2 critic steps
            if i % 2 == 0:
                g_loss = -critic(gen(gen.sample_latent(real.size(0), device))).mean()
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                g_sum += g_loss.item()
        print("Epoch %d/%d  critic=%.3f  generator=%.3f"
              % (epoch, a.epochs, c_sum / len(loader), g_sum / max(1, len(loader) // 2)))

    torch.save(gen.state_dict(), a.checkpoint)
    print("Saved", os.path.abspath(a.checkpoint))


def generate(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator().to(device)
    if not os.path.isfile(a.checkpoint):
        raise SystemExit("Checkpoint '%s' not found. Run 'python musegan.py train' first." % a.checkpoint)
    gen.load_state_dict(torch.load(a.checkpoint, map_location=device))
    gen.eval()
    os.makedirs(a.out_dir, exist_ok=True)
    with torch.no_grad():
        rolls = gen(gen.sample_latent(a.num, device)).cpu().numpy()
    for i, roll in enumerate(rolls):
        path = os.path.join(a.out_dir, "generated_%d.mid" % (i + 1))
        to_midi(roll, path, a.threshold)
        print("Wrote", path)


def main():
    p = argparse.ArgumentParser(description="Simplified MuseGAN")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--samples", type=int, default=512, help="synthetic training songs")
    t.add_argument("--data", help=".npy file shaped (N,%d,%d,%d,%d)" % (N_TRACKS, N_BARS, STEPS, PITCHES))
    t.add_argument("--checkpoint", default="musegan_generator.pth")
    t.set_defaults(fn=train)
    g = sub.add_parser("generate")
    g.add_argument("--num", type=int, default=3)
    g.add_argument("--threshold", type=float, default=0.5)
    g.add_argument("--checkpoint", default="musegan_generator.pth")
    g.add_argument("--out-dir", default="output")
    g.set_defaults(fn=generate)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
