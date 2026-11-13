#!/usr/bin/env bash
# One-time setup: installs what the app needs (ffmpeg, uv and the Python libraries, Rust), builds it, and on a Mac
# makes "Dopplero.app" to double-click (on Linux: an entry in the applications menu). Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
say() { printf '\n==> %s\n' "$1"; }

if [ "$(uname)" = Darwin ]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    say "Installing Apple's command line tools (a window opens): run ./setup.sh again when they are installed"
    xcode-select --install || true
    exit 1
  fi
  if ! command -v brew >/dev/null; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)" || true; done
  fi
  if ! command -v ffmpeg >/dev/null; then
    if ! command -v brew >/dev/null; then
      say "Installing Homebrew (it may ask for your password)"
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$("$b" shellenv)" || true; done
    fi
    say "Installing ffmpeg"
    brew install ffmpeg
  fi
elif ! command -v ffmpeg >/dev/null || ! command -v cc >/dev/null; then
  if command -v apt-get >/dev/null; then
    say "Installing ffmpeg and build tools"
    sudo apt-get update && sudo apt-get install -y ffmpeg build-essential curl
  else
    echo "Please install ffmpeg, a C compiler and curl with your package manager (e.g. dnf or pacman), then run ./setup.sh again."
    exit 1
  fi
fi

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  say "Installing uv (Python and its libraries)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! command -v cargo >/dev/null; then
  say "Installing Rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

say "Installing the Python libraries"
uv sync

say "Building the app (the first time takes a few minutes)"
DOPPLER_HOME="$ROOT" cargo build --release --manifest-path native/Cargo.toml

if [ "$(uname)" = Darwin ]; then
  APP="$ROOT/Dopplero.app"
  rm -rf "$APP"
  mkdir -p "$APP/Contents/MacOS"
  cp native/target/release/dopplero "$APP/Contents/MacOS/Dopplero"
  cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Dopplero</string>
  <key>CFBundleDisplayName</key><string>Dopplero</string>
  <key>CFBundleExecutable</key><string>Dopplero</string>
  <key>CFBundleIdentifier</key><string>local.dopplero</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
  say "Done. Double-click \"Dopplero.app\" in this folder (you can drag it to the Dock)."
  [ -t 1 ] && open -R "$APP" || true
else
  # an entry in the desktop's applications menu, like any installed app
  DESKTOP="$HOME/.local/share/applications/dopplero.desktop"
  mkdir -p "$(dirname "$DESKTOP")"
  cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Dopplero
Comment=Find where a recording was made from the planes heard in it
Exec="$ROOT/native/target/release/dopplero"
Path=$ROOT
Terminal=false
Categories=Science;AudioVideo;
EOF
  say "Done. Open \"Dopplero\" from your applications menu, or run: $ROOT/native/target/release/dopplero"
fi
