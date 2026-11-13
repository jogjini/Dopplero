//! Starts the local analysis service (tools/locate_app.py) when it isn't running yet. The service stops by itself
//! when this app exits (DOPPLER_STOP_WITH_PARENT), so nothing is left behind.
use std::{
    fs::File,
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// The project folder: the one setup.sh built the app in, else the first folder above the executable holding it.
fn root() -> Option<PathBuf> {
    let has = |p: &Path| p.join("tools/locate_app.py").is_file();
    option_env!("DOPPLER_HOME")
        .map(PathBuf::from)
        .filter(|p| has(p))
        .or_else(|| {
            std::env::current_exe()
                .ok()?
                .ancestors()
                .find(|p| has(p))
                .map(Path::to_path_buf)
        })
}

/// None when the service is already up or was started; otherwise what went wrong, for the status line.
pub fn start() -> Option<String> {
    if TcpStream::connect(("127.0.0.1", 8765)).is_ok() {
        return None;
    }
    let Some(root) = root() else {
        return Some(
            "Could not find the project folder (tools/locate_app.py): run ./setup.sh again.".into(),
        );
    };
    let python = root.join(".venv/bin/python");
    if !python.is_file() {
        return Some(format!(
            "{} is missing: run ./setup.sh first.",
            python.display()
        ));
    }
    // an app opened from Finder gets a bare PATH: ffmpeg is usually in Homebrew's folder
    let path = format!(
        "/opt/homebrew/bin:/usr/local/bin:{}",
        std::env::var("PATH").unwrap_or_default()
    );
    let log = root.join("outputs/server.log");
    let spawned = std::fs::create_dir_all(root.join("outputs"))
        .and_then(|_| File::create(&log))
        .and_then(|out| {
            Command::new(python)
                .arg("tools/locate_app.py")
                .current_dir(&root)
                .env("PATH", path)
                .env("DOPPLER_STOP_WITH_PARENT", "1")
                .stdin(Stdio::null())
                .stdout(out.try_clone()?)
                .stderr(out)
                .spawn()
        });
    spawned.err().map(|e| {
        format!(
            "Could not start the analysis service ({e}); see {}",
            log.display()
        )
    })
}
