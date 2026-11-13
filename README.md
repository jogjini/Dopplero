# Dopplero

**Find where an audio recording was made, from the planes you can hear in it.**

When a plane flies past, its engine sounds higher as it comes towards you and lower as it goes away: the Doppler
effect. On a spectrogram (a picture of a sound, time across and pitch up), each engine or propeller tone of a passing
plane draws a curve that falls from a high pitch to a low one. The shape of that curve depends on how fast the plane
was flying, how close it passed and when it was closest.

Public ADS-B archives record where almost every airliner and many small planes were, second by second. Dopplero
puts the two together. You trace the plane curves in your recording by hand, say roughly when and where it was made,
and the app tries places in that area. At each place it works out the curves the real flights would have drawn there,
and ranks the places where they fit yours best.

Why by hand? Automatic curve finders were tried during development and missed too many planes. A handful of curves
traced carefully by a person is enough to pin a recording down, as the results below show.

https://github.com/user-attachments/assets/1b3d445d-3515-48ac-9f8e-051a4cfbc419

*Demo (2:52, click to play): the audio of [a YouTube video](https://www.youtube.com/watch?v=4CxVxuMVkMc) is opened,
four plane curves are traced by hand, and a search over most of North America finds the place where the camera was.*

**Credit where it's due:** this project was inspired by [Colsto's video](https://youtu.be/UytYrhYLOKI), which came up
with the whole idea. From nothing but the sound of two propeller planes and public flight data, Colsto found where a
camera stood, within a few hundred metres, anywhere in the continental US. Dopplero turns that idea into an app
anyone can use on their own recordings.

## Results so far

These are the best searches so far, on recordings whose true location is public. The curves were marked by hand in the
app, and the search covered a huge area each time.

| Recording | Where it really was | Curves marked | Area searched | Start time known to | Aircraft checked | Best place was off by | Search time |
|---|---|---|---|---|---|---|---|
| [YouTube camera video](https://www.youtube.com/watch?v=4CxVxuMVkMc) | Long Island, New York | 4 | 7,156 × 4,399 km (most of North America) | ±5 s | 19,515 | **0.10 km** | 40 s |
| [Eclipse Soundscapes site 172](https://zenodo.org/records/22051591) | Texas | 7 | 4,876 × 2,367 km (most of the US) | ±90 s | 17,071 | **0.06 km** | 2 min 12 s |
| [Eclipse Soundscapes site 435](https://zenodo.org/records/21773341) | Texas | 4 | 4,420 × 2,181 km (most of the US) | ±2 min | 4,283 | **0.20 km** | 50 s |

**With the date only.** These searches didn't know the start time at all. They tried every start time from 00:00 to
23:00 UTC, over the whole contiguous US (5,218 × 2,923 km), against every flight of that day.

| Recording | Curves marked | Area searched | Start time known to | Best place was off by | Start time found | Search time |
|---|---|---|---|---|---|---|
| [YouTube camera video](https://www.youtube.com/watch?v=4CxVxuMVkMc) | 5 | whole contiguous US | date only | **0.07 km** | 20:17:29 UTC (true: 20:17:29) | 30 min |
| [Eclipse Soundscapes site 172](https://zenodo.org/records/22051591) | 7 | whole contiguous US | date only | **0.07 km** | 18:09:03 UTC (true: 18:09:03) | 39 min |

On site 172 the right place came first, but only narrowly ahead of the best wrong one. The app's verdict would have
said "No clear match" even though its top place was right. On the video the right place won clearly.

- **Best place was off by:** the distance from the app's number 1 place to where the recorder really was.
- **The clock was found too:** the eclipse recorders' clocks were off by up to 79 s, and the search found that error
  within 2 s.
- **Search time:** from pressing **Start search** to the result, on a MacBook Air with an Apple M4 chip (16 GB)
  using 8–10 CPU workers, with that day's flight data already downloaded. The first search on a new day also
  downloads it, which takes a few minutes or more depending on your connection.


### The recordings

- **YouTube camera video:** [youtube.com/watch?v=4CxVxuMVkMc](https://www.youtube.com/watch?v=4CxVxuMVkMc). The
  audio starts at **2026-09-08 20:17:29 UTC**.
- **Eclipse Soundscapes:** recorders left outdoors during the 8 April 2024 solar eclipse by volunteers of the
  [Eclipse Soundscapes](https://eclipsesoundscapes.org) project. Each site's recordings and published position are
  on Zenodo; download the site's `ESID_<site>_2024_04_08.zip`. The searches above used one hour of each, starting at
  these times by the recorder's own clock:


## Install

It runs on macOS (Apple Silicon or Intel) and on Linux. Windows is not supported yet. You need about 2.5 GB of disk for
the install, plus about 1 GB for each day of flight data you search. 16 GB of memory is enough for searches over a
whole continent.

### macOS

1. Download this repository (green **Code** button → **Download ZIP**, then unzip it), or in a terminal:
   `git clone https://github.com/jogjini/dopplero.git`.
2. Double-click **`Install.command`** in the folder, or run `./setup.sh` in a terminal. If macOS says it can't check
   the developer, right-click the file → **Open** → **Open**.

That one script installs everything the app needs, skipping what you already have:

- Apple's command line tools and [Homebrew](https://brew.sh), only if you don't have them. Homebrew may ask for your
  password.
- [ffmpeg](https://ffmpeg.org), which reads the audio files.
- [uv](https://docs.astral.sh/uv/), with Python 3.12 and its libraries (numpy, scipy, matplotlib, numba).
- [Rust](https://rustup.rs). The script uses it to build the desktop app, which takes a few minutes the first time.

When it finishes, the folder has a **`Dopplero.app`**. Double-click it; you can drag it to the Dock. It starts
everything it needs by itself and stops it all when you close it.

### Linux

On a desktop Linux (Ubuntu, Debian, Mint, Pop!_OS and others based on them), in a terminal:

```sh
git clone https://github.com/jogjini/dopplero.git
cd dopplero
./setup.sh
```

It installs ffmpeg and the build tools with `apt` (it asks for your password), then uv, the Python libraries and Rust,
and builds the app. When it finishes, **Dopplero** is in your applications menu. On other distributions,
install `ffmpeg`, a C compiler and `curl` with your package manager first, then run `./setup.sh`.

### Updating

Download the new version (or `git pull`) and run the install again.

## Using it

For each recording, you open it, mark its curves, set up a search, then read the results.

**1. Open a recording.** Click **+ New recording** and choose an audio or video file (mp3, m4a, wav, flac, opus,
webm, mp4…). It stays on your computer.

**2. Mark the curves.** Find the falling plane lines in the spectrogram and trace each one by hand:

- **Box a line:** drag a box around one line and the app fits a curve to it.
- **Click points along a line:** click along it, then fit.
- **Start from a curve in this view:** drag a ready-made curve onto the line.

Drag the curve or use **Snap to the line** to make it sit exactly on the line, then **Save curve**. Your curves are
saved as you go, so you can close the app and come back later.

The view buttons (**Curves**, **Propellers**, **Full range**, **Whole recording**) jump to useful settings. To move
around the spectrogram:

| To | Do |
|---|---|
| Zoom in time | Scroll |
| Zoom in pitch | Shift-scroll |
| Zoom both | Pinch, or ⌘-scroll (Ctrl-scroll on Linux) |
| Pan | Drag |
| See the whole recording | Double-click |

**3. Set up the search and start it.** In the **Search settings** step:

- **When:** the date and the start time of the recording in UTC, and how accurate that time is (for example ±5 s).
  If you only know the day, untick **I know the start time**. That works, but it is much slower.
- **Where:** drag a rectangle on the map around where it could have been. A smaller area is faster; a whole continent
  works too.
- Click **Start search**.

**4. Read the results.** The **Results** step shows the best places on the map, numbered, with the planes that explain
your curves; click a plane to see its flight. At the top is a verdict:

| Verdict | What it means |
|---|---|
| **Clear match** | The best place beats the best score found by chance by 10 or more. |
| **Possible match: check it** | The best place is ahead, but not by much. |
| **No clear match** | The places listed are no better than chance. |

These thresholds are rough rules of thumb, not calibrated probabilities. Every search is kept with the settings it
used, so you can change something and search again, or **Save result…** as a zip to share or reopen later with
**Import a saved result…**.

### Tips

- **Mark as many curves as you can**, from different planes. One curve can fit many places; four or five
  rarely agree on a wrong one.
- Trace the whole audible part of each line, especially around the steepest part of the fall: that is where the
  closest pass is.
- A start time known to a few seconds makes the search much faster and more reliable than a whole-day search.
- Only planes that broadcast ADS-B are in the flight data: nearly all airliners, many small planes, fewer helicopters
  and military aircraft. Planes that aren't in the data can't help.

## What leaves your computer

Your recordings, curves and results stay in this folder (`outputs/`). To work, the app downloads data from public
services:

- **Flight data:** a whole day of world flight data from the [adsb.lol](https://adsb.lol) archive on GitHub. Only the
  date is sent. A day is a few GB to download and about 1 GB kept in `data/adsb/world/`; delete a day's folder there
  to free the space.
- **Map tiles:** from [OpenStreetMap](https://www.openstreetmap.org), for the part of the map you are looking at.
- **Ground height and weather:** ground height from [Open-Meteo](https://open-meteo.com) (and NOAA, in the US), and
  temperature and wind from Open-Meteo, both for the area being searched.

**Use it responsibly.** Only locate recordings you have the right to locate, such as your own or ones whose place is
already public. Don't use it to find where private people live or are.

## Troubleshooting

- **"Starting the analysis service…" doesn't go away:** the log is in `outputs/server.log`. Running the install
  again usually fixes a missing library.
- **A search stopped when I closed the app:** closing the app stops a running search. Open the search and press
  **Resume**.
- **To run it from a terminal instead:** start `uv run python tools/locate_app.py`, then, in a second terminal,
  `cargo run --release --manifest-path native/Cargo.toml`.

## How the code is laid out

| Folder | What it holds |
|---|---|
| `native/` | The desktop app, in Rust with [egui](https://github.com/emilk/egui): spectrogram, the hand curve-fitting tool (`curve.rs`, `fit_panel.rs`), map and results. |
| `tools/locate_app.py` | The local analysis service the app talks to on `127.0.0.1:8765`. It renders spectrograms and runs searches. |
| `doppler/` | The analysis: audio and spectrograms (`spectrogram.py`), the physics of a passing plane (`physics.py`), flight data (`adsb/`), the place search (`trackmatch.py`, `search.py`) and a search job from start to end (`locate_job.py`). |
| `setup.sh`, `Install.command` | The install and build script, and a double-clickable way to run it on a Mac. |

This is a research project in development. The method works on the recordings it was tested on, but it isn't
guaranteed to find any given recording.

## License

[MIT](LICENSE).
