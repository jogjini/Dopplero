use serde_json::Value;
use std::{path::PathBuf, sync::mpsc::Sender, thread, time::Duration};

const BASE: &str = "http://127.0.0.1:8765";
pub enum Event {
    Json(String, Result<Value, String>),
    Image(String, Result<Vec<u8>, String>),
    Done(String, Result<(), String>),
}

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("HTTP client")
}
fn response(
    r: reqwest::Result<reqwest::blocking::Response>,
) -> Result<reqwest::blocking::Response, String> {
    let r = r.map_err(|e| format!("Could not reach the local service: {e}"))?;
    if r.status().is_success() {
        Ok(r)
    } else {
        let status = r.status();
        let message = r
            .json::<Value>()
            .ok()
            .and_then(|v| v["error"].as_str().map(str::to_owned));
        Err(message.unwrap_or_else(|| format!("Request failed ({status})")))
    }
}
pub fn get(tx: Sender<Event>, key: String, path: String) {
    thread::spawn(move || {
        let r = response(client().get(format!("{BASE}{path}")).send())
            .and_then(|r| r.json().map_err(|e| e.to_string()));
        let _ = tx.send(Event::Json(key, r));
    });
}
pub fn post(tx: Sender<Event>, key: String, path: String, body: Value) {
    thread::spawn(move || {
        let r = response(
            client()
                .post(format!("{BASE}{path}"))
                .header("X-Locate-App", "1")
                .json(&body)
                .send(),
        )
        .and_then(|r| r.json().map_err(|e| e.to_string()));
        let _ = tx.send(Event::Json(key, r));
    });
}
pub fn upload(tx: Sender<Event>, path: PathBuf, params: Value) {
    thread::spawn(move || {
        let r = (|| {
            let part = reqwest::blocking::multipart::Part::file(path).map_err(|e| e.to_string())?;
            let form = reqwest::blocking::multipart::Form::new()
                .part("audio", part)
                .text("params", params.to_string());
            response(
                client()
                    .post(format!("{BASE}/api/jobs"))
                    .header("X-Locate-App", "1")
                    .multipart(form)
                    .send(),
            )?
            .json()
            .map_err(|e| e.to_string())
        })();
        let _ = tx.send(Event::Json("created".into(), r));
    });
}
pub fn import(tx: Sender<Event>, path: PathBuf) {
    thread::spawn(move || {
        let r = (|| {
            let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            response(
                client()
                    .post(format!("{BASE}/api/import"))
                    .header("X-Locate-App", "1")
                    .body(file)
                    .send(),
            )?
            .json()
            .map_err(|e| e.to_string())
        })();
        let _ = tx.send(Event::Json("created".into(), r));
    });
}
pub fn export(tx: Sender<Event>, id: String, path: PathBuf) {
    thread::spawn(move || {
        let r = (|| {
            let mut r = response(client().get(format!("{BASE}/api/jobs/{id}/export")).send())?;
            let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
            std::io::copy(&mut r, &mut file).map_err(|e| e.to_string())?;
            Ok(())
        })();
        let _ = tx.send(Event::Done("Result saved".into(), r));
    });
}
pub fn image(tx: Sender<Event>, key: String, path: String, external: bool) {
    thread::spawn(move || {
        let url = if external {
            path
        } else {
            format!("{BASE}{path}")
        };
        let r = response(
            client()
                .get(url)
                .header("User-Agent", "Dopplero/0.1 (desktop client)")
                .send(),
        )
        .and_then(|r| r.bytes().map(|b| b.to_vec()).map_err(|e| e.to_string()));
        let _ = tx.send(Event::Image(key, r));
    });
}
