mod api;
mod curve;
mod fit_panel;
mod map;
mod server;
use api::Event;
use eframe::egui::{self, Color32, RichText, Vec2};
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    time::{Duration, Instant},
};

fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("—")
}
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
fn time(s: f64) -> String {
    let s = s.max(0.0) as u64;
    format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}
// Zoom around a point in the current interval, keeping the view within its bounds.
fn zoom_interval(
    lo: f64,
    hi: f64,
    anchor: f64,
    factor: f64,
    min: f64,
    max: f64,
    min_span: f64,
) -> (f64, f64) {
    let total = max - min;
    let span = ((hi - lo) * factor).clamp(min_span.min(total), total);
    let fraction = ((anchor - lo) / (hi - lo)).clamp(0.0, 1.0);
    let start = (anchor - fraction * span).clamp(min, max - span);
    (start, start + span)
}
/// The window one spectrogram image shows: time, frequency, axis, analysis window and pixels per second
/// (0 for a fixed strip image).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Piece {
    t0: f64,
    t1: f64,
    fmin: f64,
    fmax: f64,
    log: bool,
    win: f64,
    pps: f64,
}
/// A spectrogram image received, with its brightness per pixel (0-255, rows top = fmax).
struct Shown {
    w: Piece,
    tex: egui::TextureHandle,
    size: [usize; 2],
    lum: Vec<u8>,
}
impl Shown {
    /// Brightness (0-1) at (t, f), None outside the image.
    fn at(&self, t: f64, f: f64) -> Option<f32> {
        let w = &self.w;
        if t < w.t0 || t > w.t1 || f < w.fmin || f > w.fmax || f <= 0.0 {
            return None;
        }
        let q = if w.log {
            (f / w.fmin).ln() / (w.fmax / w.fmin).ln()
        } else {
            (f - w.fmin) / (w.fmax - w.fmin)
        };
        let x = (((t - w.t0) / (w.t1 - w.t0)) * self.size[0] as f64) as usize;
        let y = ((1.0 - q) * self.size[1] as f64) as usize;
        self.lum
            .get(y.min(self.size[1] - 1) * self.size[0] + x.min(self.size[0] - 1))
            .map(|&v| v as f32 / 255.0)
    }
}
// shortcut: up to ~2 MB of textures each, kept for the open job only; raise if long sessions keep re-rendering
const MAX_PIECES: usize = 40;
impl Piece {
    /// This image shows all of `view`, with the same axis and window, at least ~70 % of its sharpness.
    fn covers(&self, view: &Piece) -> bool {
        self.pps > 0.0
            && self.log == view.log
            && self.win == view.win
            && self.t0 <= view.t0 + 1e-6
            && self.t1 >= view.t1 - 1e-6
            && self.fmin <= view.fmin * (1.0 + 1e-9)
            && self.fmax >= view.fmax * (1.0 - 1e-9)
            && self.pps >= 0.7 * view.pps
    }
}
fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}
fn start_utc([y, mo, d, h, mi, s]: [i64; 6]) -> String {
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
/// "2026-11-11T09:40:00Z" (or with an offset after the seconds) back into its six numbers.
fn parse_utc(t: &str) -> Option<[i64; 6]> {
    let n = |a: usize, b: usize| t.get(a..b)?.parse().ok();
    Some([
        n(0, 4)?,
        n(5, 7)?,
        n(8, 10)?,
        n(11, 13)?,
        n(14, 16)?,
        n(17, 19)?,
    ])
}
// Today (UTC) at 12:00:00, the browser form's default (civil date from days since 1970: H. Hinnant's algorithm).
fn today_noon() -> [i64; 6] {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64 / 86400)
        + 719468;
    let era = days.div_euclid(146097);
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    [yoe + era * 400 + i64::from(m <= 2), m, d, 12, 0, 0]
}
fn main() -> eframe::Result {
    let problem = server::start();
    eframe::run_native(
        "Dopplero",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1450.0, 920.0])
                .with_min_inner_size([900.0, 650.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            apply_theme(&cc.egui_ctx);
            let mut app = App::new();
            app.status = problem.unwrap_or_default();
            Ok(Box::new(app))
        }),
    )
}

// A quiet, high-contrast workspace: warm paper, ink, and a single teal accent.
fn apply_theme(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(bytes) = std::fs::read("/System/Library/Fonts/HelveticaNeue.ttc") {
        fonts.font_data.insert(
            "system-ui".into(),
            std::sync::Arc::new(egui::FontData::from_owned(bytes)),
        );
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "system-ui".into());
    }
    ctx.set_fonts(fonts);
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (egui::TextStyle::Heading, egui::FontId::proportional(21.0)),
        (egui::TextStyle::Body, egui::FontId::proportional(14.0)),
        (egui::TextStyle::Button, egui::FontId::proportional(14.0)),
        (egui::TextStyle::Small, egui::FontId::proportional(12.0)),
        (egui::TextStyle::Monospace, egui::FontId::monospace(12.0)),
    ]
    .into();
    style.spacing.item_spacing = Vec2::new(10.0, 8.0);
    style.spacing.button_padding = Vec2::new(13.0, 8.0);
    style.spacing.interact_size.y = 32.0;
    style.spacing.indent = 16.0;
    let mut v = egui::Visuals::light();
    v.panel_fill = Color32::from_rgb(246, 247, 245);
    v.window_fill = Color32::WHITE;
    v.extreme_bg_color = Color32::WHITE;
    v.faint_bg_color = Color32::from_rgb(241, 245, 243);
    v.hyperlink_color = Color32::from_rgb(16, 113, 105);
    v.selection.bg_fill = Color32::from_rgb(210, 235, 229);
    v.selection.stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(17, 108, 99));
    v.window_corner_radius = 12.into();
    v.menu_corner_radius = 10.into();
    for widget in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        widget.corner_radius = 7.into();
        widget.fg_stroke.color = Color32::from_rgb(35, 52, 50);
        widget.bg_stroke = egui::Stroke::new(1.0_f32, Color32::from_rgb(218, 226, 222));
    }
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0_f32, Color32::from_rgb(229, 234, 231));
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(250, 252, 250);
    v.widgets.inactive.bg_fill = Color32::WHITE;
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(231, 242, 237);
    v.widgets.hovered.bg_fill = Color32::from_rgb(231, 242, 237);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(211, 234, 226);
    v.widgets.active.bg_fill = Color32::from_rgb(211, 234, 226);
    style.visuals = v;
    ctx.set_style(style);
}

/// The three steps of working on a recording.
#[derive(Clone, Copy, PartialEq)]
enum Step {
    Curves,
    Settings,
    Results,
}
struct App {
    tx: Sender<Event>,
    rx: Receiver<Event>,
    jobs: Value,
    /// The open recording's own job (its audio, spectrogram and curves) and its id.
    job: Value,
    current: String,
    step: Step,
    /// The search shown in Results (a job located with the recording's curves; for an old job that located
    /// directly, the recording itself).
    run: String,
    run_job: Value,
    last_run: Instant,
    /// The search the results map was last centred on (each opens on its best place once).
    focused: String,
    summary: Value,
    detections: Value,
    strips: Value,
    params: Value,
    fit: fit_panel::FitTool,
    track: Value,
    map: map::Map,
    file: Option<PathBuf>,
    /// Recording start, UTC: year, month, day, hour, minute, second.
    start: [i64; 6],
    /// Only the date is known: any start that UTC day is searched.
    date_only: bool,
    clock: f64,
    /// The search the settings form was filled from ("" = defaults).
    form_from: String,
    new_open: bool,
    search: String,
    disk: bool,
    workers: i64,
    fake: bool,
    selected: usize,
    status: String,
    confirm_stop: bool,
    busy: bool,
    last_jobs: Instant,
    last_job: Instant,
    loaded: String,
    t0: f64,
    t1: f64,
    fmin: f64,
    fmax: f64,
    log_axis: bool,
    win: f64,
    render_due: Option<Instant>,
    render_seq: u64,
    rendering: bool,
    render_inflight: Option<u64>,
    /// The window of the render in flight, to place its image where it belongs even if the view moved since.
    requested: Option<Piece>,
    /// Every spectrogram image received for this job, drawn where it belongs: seen parts stay while panning.
    pieces: Vec<Shown>,
    /// Where the spectrogram was drawn last frame (the curve tool's panel is laid out before it).
    view: Option<fit_panel::View>,
    strip_index: usize,
    strip_pending: bool,
}
impl App {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            jobs: json!([]),
            job: Value::Null,
            current: String::new(),
            step: Step::Curves,
            run: String::new(),
            run_job: Value::Null,
            last_run: Instant::now() - Duration::from_secs(20),
            focused: String::new(),
            summary: Value::Null,
            detections: json!([]),
            strips: json!([]),
            params: Value::Null,
            fit: Default::default(),
            track: Value::Null,
            map: Default::default(),
            file: None,
            start: today_noon(),
            date_only: false,
            clock: 15.0,
            // the best we have: curves fitted by hand, located by their shapes against the real flight paths
            form_from: String::new(),
            new_open: false,
            search: "tracks".into(),
            disk: true,
            workers: std::thread::available_parallelism()
                .map(|n| n.get().min(4) as i64)
                .unwrap_or(4),
            fake: false,
            selected: 0,
            status: String::new(),
            confirm_stop: false,
            busy: false,
            last_jobs: Instant::now() - Duration::from_secs(20),
            last_job: Instant::now() - Duration::from_secs(20),
            loaded: String::new(),
            t0: 0.0,
            t1: 900.0,
            fmin: 40.0,
            fmax: 800.0,
            log_axis: false,
            win: 1.0,
            render_due: None,
            render_seq: 0,
            rendering: false,
            render_inflight: None,
            requested: None,
            pieces: vec![],
            view: None,
            strip_index: 0,
            strip_pending: false,
        }
    }
    fn get(&self, key: &str, path: String) {
        api::get(self.tx.clone(), key.into(), path);
    }
    fn open(&mut self, id: String) {
        self.current = id;
        self.job = Value::Null;
        self.summary = Value::Null;
        self.detections = json!([]);
        self.strips = json!([]);
        self.params = Value::Null;
        self.fit = Default::default();
        self.track = Value::Null;
        self.loaded.clear();
        self.pieces.clear();
        self.requested = None;
        self.strip_pending = false;
        self.render_seq += 1;
        self.rendering = false;
        self.render_inflight = None;
        self.render_due = None;
        self.last_job = Instant::now() - Duration::from_secs(10);
        self.selected = 0;
        self.status.clear();
        // its latest search shown, the form filled from it; a recording not searched yet opens on its curves
        let latest = self.searches(&self.current).first().cloned();
        self.select_run(latest.as_ref().map_or("", |r| text(&r["id"])).to_owned());
        self.step = if latest.is_some() {
            Step::Results
        } else {
            Step::Curves
        };
        if let Some(r) = latest {
            self.fill_form(&r);
        }
        self.map.drawing = self.map.bounds.is_none();
    }
    fn select_run(&mut self, id: String) {
        self.run = id;
        self.run_job = Value::Null;
        self.track = Value::Null;
        self.selected = 0;
        self.last_run = Instant::now() - Duration::from_secs(10);
    }
    /// The searches of a recording, newest first: the jobs located with its curves, and itself if it located.
    fn searches(&self, root: &str) -> Vec<Value> {
        self.jobs
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|j| {
                        text(&j["root"]) == root
                            && (text(&j["id"]) != root || j["detect_only"] != true)
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
    /// The settings form from a search's own settings (from the job list), to see them or change and search again.
    fn fill_form(&mut self, row: &Value) {
        let s = &row["settings"];
        if let Some(t) = s["start_utc"].as_str().and_then(parse_utc) {
            self.start = t;
        }
        self.date_only = s["date_only"] == true;
        if let Some(c) = s["clock_s"].as_f64() {
            self.clock = c;
        }
        if let Some(b) = s["box"].as_array().filter(|b| b.len() == 4) {
            let b: Vec<f64> = b.iter().map(number).collect();
            self.map.bounds = Some([b[0], b[1], b[2], b[3]]);
        }
        if let Some(m) = s["search"].as_str() {
            self.search = m.into();
        }
        if let Some(w) = s["workers"].as_i64() {
            self.workers = w;
        }
        self.fake = s["fake_sky"] == true;
        self.disk = s["data_mode"] != "ram";
        self.form_from = text(&row["id"]).into();
    }
    fn load_files(&self) {
        for name in ["summary", "detections", "spectrograms", "params", "manual"] {
            self.get(
                &format!("file:{}:{name}", self.current),
                format!("/api/jobs/{}/files/{name}.json", self.current),
            );
        }
    }
    fn params(&self) -> Value {
        // everything else (wind, temperature, heights, thresholds) is the pipeline's tested defaults
        let [y, mo, d, ..] = self.start;
        let start = if self.date_only {
            start_utc([y, mo, d, 0, 0, 0])
        } else {
            start_utc(self.start)
        };
        json!({"start_utc":start,"date_only":self.date_only,"clock_s":self.clock,"box":self.map.bounds,"detect_only":false,"search":self.search,"data_mode":if self.disk {"disk"} else {"ram"},"workers":self.workers,"fake_sky":self.fake})
    }
    fn schedule(&mut self) {
        self.render_seq += 1;
        self.render_due = Some(Instant::now() + Duration::from_millis(250));
        self.rendering = true;
    }
    fn events(&mut self, ctx: &egui::Context) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                Event::Json(key, r) => match r {
                    Ok(v) => {
                        if key == "jobs" {
                            self.jobs = v;
                            if self.status.starts_with("Starting the analysis service") {
                                self.status.clear();
                            }
                        } else if key == format!("job:{}", self.current) {
                            let state = text(&v["status"]).to_owned();
                            let ready = v["result"].is_object();
                            self.job = v;
                            if ready && self.loaded != self.current {
                                self.loaded = self.current.clone();
                                self.load_files();
                            }
                            if state == "running" && self.summary.is_null() {
                                self.load_files();
                            }
                        } else if key == format!("run:{}", self.run) {
                            self.run_job = v;
                        } else if key == "created" {
                            self.busy = false;
                            if let Some(id) = v["id"].as_str() {
                                self.open(id.into());
                                self.status = "Recording added: it opens here once loaded.".into();
                                self.last_jobs = Instant::now() - Duration::from_secs(20);
                            }
                        } else if key == "searched" {
                            self.busy = false;
                            if let Some(id) = v["id"].as_str() {
                                self.select_run(id.into());
                                self.step = Step::Results;
                                self.status.clear();
                                self.last_jobs = Instant::now() - Duration::from_secs(20);
                            }
                        } else if key.starts_with(&format!("file:{}:", self.current)) {
                            let name = key.rsplit(':').next().unwrap_or("");
                            match name {
                                "summary" => {
                                    let d = number(&v["duration_s"]);
                                    if self.summary.is_null() && d > 0.0 {
                                        self.t0 = 0.0;
                                        self.t1 = d.min(900.0);
                                        if self.job["has_audio"] == true {
                                            self.schedule();
                                        }
                                    }
                                    self.summary = v;
                                }
                                "detections" => self.detections = v,
                                "spectrograms" => self.strips = v,
                                "params" => self.params = v,
                                "manual" => self.fit.load(&v),
                                _ => {}
                            }
                        } else if key == format!("track:{}", self.run) {
                            self.track = v;
                        } else if key.starts_with("manual:") {
                            self.status =
                                format!("Hand-fitted curves saved ({})", text_num(&v["saved"]));
                        } else if key == "stop" || key == "rerun" {
                            self.busy = false;
                            self.last_run = Instant::now() - Duration::from_secs(10);
                            self.last_jobs = Instant::now() - Duration::from_secs(20);
                        }
                    }
                    Err(e) => {
                        if key.starts_with("manual:") {
                            self.status = format!("Could not save the hand-fitted curves: {e}");
                        } else if key == "created"
                            || key == "searched"
                            || key == "stop"
                            || key == "rerun"
                        {
                            self.busy = false;
                            self.status = e;
                        } else if key == "jobs" {
                            if self.status.is_empty() || self.status.starts_with("Starting") {
                                // the service takes a few seconds to start; the list is asked for again until it answers
                                self.status = "Starting the analysis service… (if this stays, see outputs/server.log)".into();
                            }
                        } else if key.starts_with("track:") {
                            self.status = e;
                        }
                    }
                },
                Event::Image(key, r) => {
                    if key.starts_with("tile/") {
                        self.map.tile(ctx, key, r);
                    } else {
                        // a render answered is kept even when the view has moved on: it is placed by its own window
                        let render = key.starts_with(&format!("render:{}:", self.current))
                            && key.rsplit(':').next().and_then(|v| v.parse::<u64>().ok())
                                == self.render_inflight;
                        let strip = key == format!("strip:{}:{}", self.current, self.strip_index);
                        let window = if render {
                            self.render_inflight = None;
                            self.rendering = self.render_due.is_some();
                            self.requested.take()
                        } else if strip {
                            self.pieces.clear();
                            // no panning without audio: the view is still the strip's
                            Some(Piece {
                                t0: self.t0,
                                t1: self.t1,
                                fmin: self.fmin,
                                fmax: self.fmax,
                                log: self.log_axis,
                                win: self.win,
                                pps: 0.0,
                            })
                        } else {
                            None
                        };
                        if let Some(window) = window {
                            match r.and_then(|b| {
                                image::load_from_memory(&b).map_err(|e| e.to_string())
                            }) {
                                Ok(im) => {
                                    let im = im.to_rgba8();
                                    let tex = ctx.load_texture(
                                        "spectrogram",
                                        egui::ColorImage::from_rgba_unmultiplied(
                                            [im.width() as usize, im.height() as usize],
                                            im.as_raw(),
                                        ),
                                        Default::default(),
                                    );
                                    // brightness per pixel (inferno's lightness rises with level), for snapping curves
                                    let lum = im
                                        .pixels()
                                        .map(|q| {
                                            (0.2126 * q[0] as f32
                                                + 0.7152 * q[1] as f32
                                                + 0.0722 * q[2] as f32)
                                                as u8
                                        })
                                        .collect();
                                    self.pieces.push(Shown {
                                        w: window,
                                        tex,
                                        size: [im.width() as usize, im.height() as usize],
                                        lum,
                                    });
                                    if self.pieces.len() > MAX_PIECES {
                                        drop(self.pieces.remove(0)); // the oldest
                                    }
                                }
                                Err(e) => self.status = format!("Could not load spectrogram: {e}"),
                            }
                            self.strip_pending = !strip && self.strip_pending;
                        }
                    }
                }
                Event::Done(label, r) => {
                    self.busy = false;
                    self.status = r.err().unwrap_or(label);
                }
            }
        }
    }
    fn send_manual(&self) {
        api::post(
            self.tx.clone(),
            format!("manual:{}", self.current),
            format!("/api/jobs/{}/manual", self.current),
            json!({"curves": self.fit.to_json()}),
        );
    }
    /// "New recording": choose the file. Opening it only loads the audio; time,
    /// area and search are set later, in Search settings.
    fn new_recording_ui(&mut self, ctx: &egui::Context) {
        let mut open = self.new_open;
        egui::Window::new("New recording")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_width(420.0);
                ui.label(RichText::new("Use only on recordings you have the right to locate, such as public recordings whose location is already published.").small().color(Color32::from_rgb(132, 91, 39)));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Choose audio or video…").clicked() {
                        if let Some(f) = rfd::FileDialog::new().pick_file() {
                            self.file = Some(f);
                        }
                    }
                    ui.label(
                        self.file
                            .as_ref()
                            .and_then(|p| p.file_name())
                            .map_or("No file chosen".into(), |p| p.to_string_lossy().into_owned()),
                    );
                });
                ui.label(RichText::new("Next you mark the planes' curves on its spectrogram; when and where it was recorded is asked after that.").small().weak());
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let ready = self.file.is_some() && !self.busy;
                    if ui
                        .add_enabled(ready, egui::Button::new(RichText::new("Open recording").strong().color(Color32::WHITE)).fill(ACCENT))
                        .clicked()
                    {
                        if let Some(path) = self.file.clone() {
                            self.busy = true;
                            self.new_open = false;
                            api::upload(
                                self.tx.clone(),
                                path,
                                json!({"detect_only": true, "search": "tracks", "workers": self.workers, "data_mode": "disk"}),
                            );
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.new_open = false;
                    }
                });
            });
        self.new_open &= open;
    }
    /// Search settings: when, where, which curves, and the advanced options; one button starts the search.
    fn settings_step(&mut self, ui: &mut egui::Ui) {
        let n = self.fit.saved.len();
        let ready = !self.summary.is_null();
        let blocker = if !ready {
            Some("The recording is still loading.")
        } else if n == 0 {
            Some("Mark at least one curve first (step 1).")
        } else if self.map.bounds.is_none() {
            Some("Draw the search area on the map.")
        } else {
            None
        };
        egui::SidePanel::left("search_form")
            .resizable(false)
            .exact_width(390.0)
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(4, 8)))
            .show_inside(ui, |ui| {
                egui::TopBottomPanel::bottom("start_bar")
                    .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 10)))
                    .show_inside(ui, |ui| {
                        let start = ui.add_enabled(
                            blocker.is_none() && !self.busy,
                            egui::Button::new(RichText::new("Start search  →").strong().size(16.0).color(Color32::WHITE))
                                .fill(ACCENT)
                                .min_size(Vec2::new(ui.available_width(), 40.0)),
                        );
                        if start.clicked() {
                            self.busy = true;
                            api::post(
                                self.tx.clone(),
                                "searched".into(),
                                format!("/api/jobs/{}/relocate", self.current),
                                json!({"settings": self.params()}),
                            );
                        }
                        ui.label(
                            RichText::new(blocker.unwrap_or(if self.date_only {
                                "Searching a whole day takes tens of minutes, more on a large area."
                            } else {
                                "Results appear in step 3 as the search runs."
                            }))
                            .small()
                            .weak(),
                        );
                    });
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(i) = self
                        .searches(&self.current)
                        .iter()
                        .rev()
                        .position(|r| text(&r["id"]) == self.form_from)
                    {
                        ui.label(
                            RichText::new(format!("Filled in from search #{}. Change anything, then start a new search.", i + 1))
                                .small()
                                .weak(),
                        );
                    }
                    ui.add_space(4.0);
                    ui.strong("1  When was it recorded? (UTC)");
                    ui.horizontal(|ui| {
                        let [y, mo, d, h, mi, s] = &mut self.start;
                        ui.add(egui::DragValue::new(y).range(1990..=2100)).on_hover_text("Year");
                        ui.label("-");
                        ui.add(egui::DragValue::new(mo).range(1..=12)).on_hover_text("Month");
                        ui.label("-");
                        let last = days_in_month(*y, *mo);
                        ui.add(egui::DragValue::new(d).range(1..=last)).on_hover_text("Day");
                        *d = (*d).clamp(1, last); // e.g. 31 kept when the month changes to April
                        if !self.date_only {
                            ui.add_space(8.0);
                            ui.add(egui::DragValue::new(h).range(0..=23)).on_hover_text("Hour");
                            ui.label(":");
                            ui.add(egui::DragValue::new(mi).range(0..=59)).on_hover_text("Minute");
                            ui.label(":");
                            ui.add(egui::DragValue::new(s).range(0..=59)).on_hover_text("Second");
                        }
                    });
                    let mut known = !self.date_only;
                    ui.checkbox(&mut known, "I know the start time");
                    self.date_only = !known;
                    if known {
                        ui.horizontal(|ui| {
                            ui.label("Clock accurate to ±");
                            ui.add(egui::DragValue::new(&mut self.clock).range(0.0..=86400.0).suffix(" s"));
                        });
                        ui.label(RichText::new("The time of the recording's first second, in UTC (not local time).").small().weak());
                    } else {
                        ui.label(RichText::new("Every start time that UTC day is tried. Note a local evening can be the next day in UTC.").small().weak());
                    }
                    ui.add_space(10.0);
                    ui.strong("2  Where could it be?");
                    match self.map.bounds {
                        Some(b) => {
                            ui.label(format!("Area of about {} drawn on the map.", area_km(&json!(b))));
                        }
                        None => {
                            ui.label("Drag a rectangle on the map to the right.");
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button(if self.map.drawing { "Drawing… (drag on the map)" } else { "Draw a new area" }).clicked() {
                            self.map.drawing = !self.map.drawing;
                        }
                        if self.map.bounds.is_some() && ui.button("Clear").clicked() {
                            self.map.bounds = None;
                            self.map.drawing = true;
                        }
                    });
                    ui.label(RichText::new("Smaller is faster and more reliable: a whole country works, but takes much longer.").small().weak());
                    ui.add_space(10.0);
                    ui.strong("3  Curves");
                    ui.label(format!("{n} curve(s) will be used."));
                    ui.add_space(10.0);
                    egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
                        egui::ComboBox::from_label("Method")
                            .selected_text(method_name(&self.search))
                            .show_ui(ui, |ui| {
                                for id in ["tracks", "forward", "fits"] {
                                    ui.selectable_value(&mut self.search, id.into(), method_name(id));
                                }
                            });
                        ui.horizontal(|ui| {
                            ui.label("CPU workers");
                            ui.add(egui::DragValue::new(&mut self.workers).range(1..=64));
                        });
                        ui.label(RichText::new("4 is safe on a 16 GB Mac; the search gives workers back if memory runs low.").small().weak());
                        ui.checkbox(&mut self.fake, "Also search a fake sky (time-shifted flights)");
                        ui.label(RichText::new("Shows what scores by chance; doubles the time. A whole-day search measures chance anyway.").small().weak());
                        ui.checkbox(&mut self.disk, "Keep flight data on disk for reuse");
                    });
                });
            });
        ui.add_space(4.0);
        ui.label(
            RichText::new(if self.map.drawing {
                "Drag on the map to draw the search area"
            } else {
                "Drag to pan · scroll to zoom"
            })
            .small()
            .weak(),
        );
        self.map.ui(ui, &self.tx, &Value::Null, &Value::Null);
    }
    /// The spectrogram with its view controls, filling the space it is given; clicks and drags go to the curve tool.
    fn recording_ui(&mut self, ui: &mut egui::Ui) {
        let audio = self.job["has_audio"].as_bool().unwrap_or(false);
        let duration = number(&self.summary["duration_s"]);
        let mut changed = false;
        if audio {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("View").strong());
                for (label, lo, hi, log) in [
                    ("Curves", 40.0, 800.0, false),
                    ("Propellers", 30.0, 600.0, false),
                    ("Full range", 30.0, 3000.0, true),
                ] {
                    if ui.button(label).clicked() {
                        (self.fmin, self.fmax, self.log_axis, self.win) = (lo, hi, log, 1.0);
                        changed = true;
                    }
                }
                if ui.button("Whole recording").clicked() {
                    self.t0 = 0.0;
                    self.t1 = duration.min(3600.0);
                    changed = true;
                }
                ui.separator();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.fmin)
                            .range(15.0..=7499.0)
                            .suffix(" Hz"),
                    )
                    .changed();
                ui.label("to");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.fmax)
                            .range(16.0..=7500.0)
                            .suffix(" Hz"),
                    )
                    .changed();
                changed |= ui.checkbox(&mut self.log_axis, "Log scale").changed();
                egui::ComboBox::from_label("Window")
                    .selected_text(format!("{} s", self.win))
                    .show_ui(ui, |ui| {
                        for v in [0.25, 1.0, 4.0] {
                            changed |= ui
                                .selectable_value(&mut self.win, v, format!("{v} s"))
                                .changed();
                        }
                    });
            });
            if self.fmin >= self.fmax {
                self.fmax = (self.fmin + 1.0).min(7500.0);
                self.fmin = self.fmin.min(self.fmax - 1.0);
            }
            if changed {
                self.schedule();
            }
        } else {
            ui.label("This saved result has no audio. Fixed spectrogram images are shown instead.");
            if let Some(strips) = self.strips.as_array() {
                let n = strips.len();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.strip_index > 0, egui::Button::new("← Previous strip"))
                        .clicked()
                    {
                        self.strip_index -= 1;
                        self.pieces.clear();
                        self.strip_pending = false;
                    }
                    ui.label(format!("Strip {} of {}", self.strip_index + 1, n));
                    if ui
                        .add_enabled(self.strip_index + 1 < n, egui::Button::new("Next strip →"))
                        .clicked()
                    {
                        self.strip_index += 1;
                        self.pieces.clear();
                        self.strip_pending = false;
                    }
                });
                if self.pieces.is_empty() && !self.strip_pending && self.strip_index < n {
                    let s = &strips[self.strip_index];
                    if let Some(file) = s["file_lin"].as_str().or_else(|| s["file"].as_str()) {
                        self.t0 = number(&s["t0_s"]);
                        self.t1 = number(&s["t1_s"]);
                        if s["file_lin"].is_string() {
                            self.fmin = s["lin_hz"][0].as_f64().unwrap_or(30.0);
                            self.fmax = s["lin_hz"][1].as_f64().unwrap_or(600.0);
                            self.log_axis = false;
                        } else {
                            let base = s["fmin_hz"].as_f64().unwrap_or(30.0);
                            let bins = s["bins_per_octave"].as_f64().unwrap_or(1.0).max(1.0);
                            self.fmin = base * 2_f64.powf(number(&s["bin_lo"]) / bins);
                            self.fmax = base * 2_f64.powf(number(&s["bin_hi"]) / bins);
                            self.log_axis = true;
                        }
                        self.strip_pending = true;
                        api::image(
                            self.tx.clone(),
                            format!("strip:{}:{}", self.current, self.strip_index),
                            format!("/api/jobs/{}/files/{file}", self.current),
                            false,
                        );
                    }
                }
            }
        }
        let width = ui.available_width().max(200.0);
        // the rest of the page: room left for the overview strip and the hint
        let height = (ui.available_height() - 70.0).max(220.0);
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::click_and_drag());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 4.0, Color32::from_rgb(19, 18, 29));
        let view = fit_panel::View {
            rect,
            t0: self.t0,
            t1: self.t1,
            fmin: self.fmin,
            fmax: self.fmax,
            log: self.log_axis,
        };
        self.view = Some(view);
        // every image of this axis and window, coarse ones first so the sharpest ends on top; the newest wins a tie
        let mut shown: Vec<&Shown> = self
            .pieces
            .iter()
            .filter(|q| {
                q.w.log == self.log_axis
                    && (q.w.win == self.win || q.w.pps == 0.0)
                    && q.w.t1 > self.t0
                    && q.w.t0 < self.t1
            })
            .collect();
        shown.sort_by(|a, b| a.w.pps.total_cmp(&b.w.pps));
        for Shown { w, tex, .. } in shown {
            if let (Some(a), Some(b)) = (view.pos(w.t0, w.fmax), view.pos(w.t1, w.fmin)) {
                p.image(
                    tex.id(),
                    egui::Rect::from_two_pos(a, b),
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }
        let pointer = response.hover_pos().filter(|pos| rect.contains(*pos));
        // the curves of a saved result (its marked curves come as detections): drawn, not edited
        if let Some(dets) = self.detections.as_array() {
            for c in dets
                .iter()
                .filter_map(|d| d["contours"].as_array())
                .flatten()
            {
                if let (Some(ts), Some(fs)) = (c["t_s"].as_array(), c["f_hz"].as_array()) {
                    let points: Vec<_> = ts
                        .iter()
                        .zip(fs)
                        .filter_map(|(t, f)| view.pos(t.as_f64()?, f.as_f64()?))
                        .collect();
                    if points.len() > 1 {
                        p.add(egui::Shape::line(
                            points,
                            egui::Stroke::new(1.5_f32, Color32::from_rgb(125, 200, 236)),
                        ));
                    }
                }
            }
        }
        self.fit.draw(&p, &view);
        let mods = ui.input(|i| i.modifiers);
        let boxing = (self.fit.boxer.on || self.fit.erasing) && self.fit.active.is_some();
        if boxing && pointer.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        } else if self.fit.dragging || pointer.is_some_and(|pos| self.fit.near_curve(&view, pos)) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        } else if self.fit.active.is_some()
            && pointer.is_some()
            && (self.fit.click_adds || mods.command || mods.alt)
        {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }
        if response.clicked() && !response.double_clicked() {
            // with a curve being fitted, clicks place its points (Alt removes one)
            if let Some(pos) = response.interact_pointer_pos() {
                self.fit.click(&view, pos, mods);
            }
        }
        if response.secondary_clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.fit.remove_near(&view, pos);
            }
        }
        // a drag draws a box when boxing; else one that starts on the curve being fitted moves the curve; any other scrolls
        if response.drag_started() {
            let origin = ui.input(|i| i.pointer.press_origin());
            if boxing {
                self.fit.boxer.from = origin.map(|pos| view.at(pos));
            } else {
                self.fit.dragging = origin.is_some_and(|pos| self.fit.near_curve(&view, pos));
            }
        }
        let latest = ui.input(|i| i.pointer.latest_pos());
        if let (Some(a), Some(b)) = (
            self.fit.boxer.from.and_then(|(t, f)| view.pos(t, f)),
            latest,
        ) {
            let r = egui::Rect::from_two_pos(a, b.clamp(rect.min, rect.max));
            let color = if self.fit.erasing {
                Color32::from_rgb(255, 90, 90)
            } else {
                Color32::WHITE
            };
            p.rect_stroke(
                r,
                0.0,
                egui::Stroke::new(1.5_f32, color),
                egui::StrokeKind::Middle,
            );
        }
        if self.fit.dragging && response.dragged() {
            if let Some(pos) = response.interact_pointer_pos() {
                let d = ui.input(|i| i.pointer.delta());
                self.fit.drag(&view, pos - d, pos);
            }
        }
        if response.drag_stopped() {
            self.fit.dragging = false;
            if let (Some((t0, f0)), Some(b)) = (self.fit.boxer.from.take(), latest) {
                let (t1, f1) = view.at(b.clamp(rect.min, rect.max));
                if self.fit.erasing {
                    self.fit.erase(((t0, t1), (f0, f1)));
                } else {
                    self.fit.boxer.drawn = Some(((t0, t1), (f0, f1)));
                }
            }
        }
        p.text(
            rect.left_top() + Vec2::new(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            format!("{:.0} Hz", self.fmax),
            egui::FontId::proportional(11.0),
            Color32::WHITE,
        );
        p.text(
            rect.left_bottom() + Vec2::new(8.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{}  ·  {:.0} Hz", time(self.t0), self.fmin),
            egui::FontId::proportional(11.0),
            Color32::WHITE,
        );
        p.text(
            rect.right_bottom() - Vec2::new(8.0, 6.0),
            egui::Align2::RIGHT_BOTTOM,
            time(self.t1),
            egui::FontId::proportional(11.0),
            Color32::WHITE,
        );
        if audio {
            if response.dragged() && !self.fit.dragging && self.fit.boxer.from.is_none() {
                let delta =
                    ui.input(|i| i.pointer.delta().x) as f64 / width as f64 * (self.t1 - self.t0);
                let span = self.t1 - self.t0;
                self.t0 = (self.t0 - delta).clamp(0.0, (duration - span).max(0.0));
                self.t1 = self.t0 + span;
                self.schedule();
            }
            if response.double_clicked() {
                self.t0 = 0.0;
                self.t1 = duration.min(3600.0);
                self.schedule();
            }
            if response.hovered() {
                // Consume the wheel here before the enclosing page sees it.
                let (scroll, shift, pointer, zoom) = ui.input_mut(|i| {
                    let delta = i.smooth_scroll_delta;
                    i.smooth_scroll_delta = Vec2::ZERO;
                    (
                        delta,
                        i.modifiers.shift,
                        i.pointer.hover_pos(),
                        i.zoom_delta(),
                    )
                });
                let pointer = pointer.unwrap_or(rect.center());
                // (factor, zoom time, zoom frequency): a pinch or Cmd/Ctrl-scroll zooms both alike, like zooming
                // into a picture (curve shapes kept); scroll zooms time, Shift-scroll frequency
                let (k, zoom_time, zoom_freq) = if (zoom - 1.0).abs() > 1e-4 {
                    (1.0 / zoom as f64, true, true)
                } else if scroll.x.abs() > scroll.y.abs() && !shift {
                    let span = self.t1 - self.t0;
                    self.t0 = (self.t0 - scroll.x as f64 / width as f64 * span)
                        .clamp(0.0, (duration - span).max(0.0));
                    self.t1 = self.t0 + span;
                    self.schedule();
                    (1.0, false, false)
                } else if scroll.y.abs() > 0.1 || (shift && scroll.x.abs() > 0.1) {
                    let delta = if scroll.y.abs() > 0.1 {
                        scroll.y
                    } else {
                        scroll.x
                    };
                    ((-delta as f64 * 0.002).exp(), !shift, shift)
                } else {
                    (1.0, false, false)
                };
                if zoom_freq {
                    let fraction = ((rect.bottom() - pointer.y) / height).clamp(0.0, 1.0) as f64;
                    if self.log_axis {
                        let lo = self.fmin.ln();
                        let hi = self.fmax.ln();
                        let (a, b) = zoom_interval(
                            lo,
                            hi,
                            lo + fraction * (hi - lo),
                            k,
                            15.0_f64.ln(),
                            7500.0_f64.ln(),
                            1.01_f64.ln(),
                        );
                        self.fmin = a.exp();
                        self.fmax = b.exp();
                    } else {
                        let anchor = self.fmin + fraction * (self.fmax - self.fmin);
                        (self.fmin, self.fmax) =
                            zoom_interval(self.fmin, self.fmax, anchor, k, 15.0, 7500.0, 1.0);
                    }
                }
                if zoom_time && duration > 0.0 {
                    let fraction = ((pointer.x - rect.left()) / width).clamp(0.0, 1.0) as f64;
                    let anchor = self.t0 + fraction * (self.t1 - self.t0);
                    // A view can cover at most an hour, even in longer recordings.
                    let span = ((self.t1 - self.t0) * k)
                        .clamp(0.1_f64.min(duration), duration.min(3600.0));
                    self.t0 = (anchor - fraction * span).clamp(0.0, (duration - span).max(0.0));
                    self.t1 = self.t0 + span;
                }
                if zoom_time || zoom_freq {
                    self.schedule();
                }
            }
        }
        if audio && duration > 0.0 {
            let (overview, response) =
                ui.allocate_exact_size(Vec2::new(width, 24.0), egui::Sense::click_and_drag());
            let painter = ui.painter_at(overview);
            painter.rect_filled(overview, 5.0, Color32::from_rgb(219, 230, 225));
            if let Some(dets) = self.detections.as_array() {
                for d in dets {
                    let x = overview.left()
                        + (number(&d["t_cpa_s"]) / duration).clamp(0.0, 1.0) as f32 * width;
                    painter.line_segment(
                        [
                            egui::pos2(x, overview.top() + 6.0),
                            egui::pos2(x, overview.bottom() - 6.0),
                        ],
                        egui::Stroke::new(1.0_f32, Color32::from_rgb(93, 136, 123)),
                    );
                }
            }
            let left = overview.left() + (self.t0 / duration) as f32 * width;
            let right = overview.left() + (self.t1 / duration) as f32 * width;
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(left, overview.top()),
                    egui::pos2(right.max(left + 3.0), overview.bottom()),
                ),
                4.0,
                Color32::from_rgba_unmultiplied(19, 112, 96, 65),
            );
            painter.rect_stroke(
                egui::Rect::from_min_max(
                    egui::pos2(left, overview.top()),
                    egui::pos2(right.max(left + 3.0), overview.bottom()),
                ),
                4.0,
                egui::Stroke::new(1.5_f32, Color32::from_rgb(16, 105, 94)),
                egui::StrokeKind::Inside,
            );
            if (response.clicked() || response.dragged())
                && response.interact_pointer_pos().is_some()
            {
                let pointer = response.interact_pointer_pos().unwrap();
                let center =
                    ((pointer.x - overview.left()) / width).clamp(0.0, 1.0) as f64 * duration;
                let span = self.t1 - self.t0;
                self.t0 = (center - span / 2.0).clamp(0.0, (duration - span).max(0.0));
                self.t1 = self.t0 + span;
                self.schedule();
            }
        }
        ui.label(
            RichText::new(if self.rendering {
                "Updating spectrogram…"
            } else {
                "Scroll: zoom time · Shift-scroll: zoom frequency · Pinch or ⌘/Ctrl-scroll: zoom both · Drag to pan · Double-click: whole recording"
            })
            .small()
            .weak(),
        );
    }
    /// The panel beside the spectrogram: the open curve's tools (Save / Cancel kept in view above them), or, with
    /// none open, how to start one and your curves.
    fn curve_panel(&mut self, ui: &mut egui::Ui) {
        let Some(view) = self.view else { return };
        if self.fit.active.is_some() {
            egui::TopBottomPanel::top("curve_head")
                .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 6)))
                .show_inside(ui, |ui| {
                    if self.fit.header_ui(ui) {
                        self.send_manual();
                    }
                });
            egui::ScrollArea::vertical()
                .id_salt("curve_edit")
                .show(ui, |ui| {
                    let pieces = &self.pieces;
                    let (log, win) = (self.log_axis, self.win);
                    let bright = |t: f64, f: f64| {
                        pieces
                            .iter()
                            .filter(|q| q.w.log == log && (q.w.win == win || q.w.pps == 0.0))
                            .filter_map(|q| Some((q.w.pps, q.at(t, f)?)))
                            .max_by(|a, b| a.0.total_cmp(&b.0))
                            .map(|(_, v)| v)
                    };
                    self.fit.edit_ui(ui, &view, &bright);
                });
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt("curve_start")
            .show(ui, |ui| {
                if self.fit.start_ui(ui, &view) {
                    self.send_manual();
                }
            });
    }
    /// Step 1: the spectrogram, the curve panel beside it, and a bar saying the curves are saved, with the way on.
    fn curves_step(&mut self, ui: &mut egui::Ui) {
        if self.summary.is_null() {
            let status = text(&self.job["status"]);
            if status == "running" || status == "queued" {
                ui.add_space(20.0);
                progress_ui(ui, &self.job);
            } else if status == "failed" {
                ui.colored_label(
                    Color32::RED,
                    format!(
                        "The recording could not be loaded: {}",
                        text(&self.job["error"])
                    ),
                );
            } else {
                ui.spinner();
            }
            return;
        }
        egui::TopBottomPanel::bottom("curves_bar")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 10)))
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let n = self.fit.saved.len();
                    ui.label(
                        RichText::new(format!(
                            "✓ {n} curve(s) · saved automatically: you can close the app and come back later"
                        ))
                        .color(Color32::from_rgb(40, 110, 80)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                n > 0,
                                egui::Button::new(RichText::new("Next: search settings  →").strong().color(Color32::WHITE)).fill(ACCENT),
                            )
                            .on_disabled_hover_text("Mark at least one curve first")
                            .clicked()
                        {
                            self.step = Step::Settings;
                        }
                    });
                });
            });
        egui::SidePanel::right("curve_tool")
            .resizable(true)
            .default_width(360.0)
            .width_range(300.0..=520.0)
            .show_inside(ui, |ui| self.curve_panel(ui));
        self.recording_ui(ui);
    }
    /// Step 3: the recording's searches (newest first) and the one selected: its verdict, actions, candidates on
    /// the map, the settings it used and its log.
    fn results_step(&mut self, ui: &mut egui::Ui) {
        let runs = self.searches(&self.current);
        if runs.is_empty() {
            ui.add_space(30.0);
            ui.heading("No searches yet");
            ui.label("Set when and where it was recorded, then start a search.");
            if ui.button("Go to search settings  →").clicked() {
                self.step = Step::Settings;
            }
            return;
        }
        if !runs.iter().any(|r| text(&r["id"]) == self.run) {
            self.select_run(text(&runs[0]["id"]).into());
        }
        let n = runs.len();
        egui::SidePanel::left("runs")
            .resizable(false)
            .exact_width(270.0)
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 6)))
            .show_inside(ui, |ui| {
                ui.strong("Searches");
                egui::ScrollArea::vertical()
                    .id_salt("runs_list")
                    .show(ui, |ui| {
                        for (i, r) in runs.iter().enumerate() {
                            let id = text(&r["id"]);
                            let on = id == self.run;
                            let resp = egui::Frame::new()
                                .fill(if on {
                                    Color32::from_rgb(226, 242, 236)
                                } else {
                                    Color32::WHITE
                                })
                                .stroke(egui::Stroke::new(
                                    1.0_f32,
                                    if on {
                                        ACCENT
                                    } else {
                                        Color32::from_rgb(226, 232, 228)
                                    },
                                ))
                                .corner_radius(8)
                                .inner_margin(egui::Margin::symmetric(10, 8))
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        ui.strong(format!("#{}", n - i));
                                        ui.label(RichText::new(started(id)).small().weak());
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                status_chip(ui, r);
                                            },
                                        );
                                    });
                                    ui.label(RichText::new(when(&r["settings"])).small());
                                    ui.label(
                                        RichText::new(format!(
                                            "area {}",
                                            area_km(&r["settings"]["box"])
                                        ))
                                        .small()
                                        .weak(),
                                    );
                                })
                                .response
                                .interact(egui::Sense::click());
                            if resp.clicked() && !on {
                                self.select_run(id.into());
                            }
                            ui.add_space(2.0);
                        }
                    });
            });
        let Some(row) = runs.iter().find(|r| text(&r["id"]) == self.run).cloned() else {
            return;
        };
        let number_of = n - runs
            .iter()
            .position(|r| text(&r["id"]) == self.run)
            .unwrap_or(0);
        // this search's own status, polled every 2 s (the list's is up to 10 s old)
        let status = self.run_job["status"]
            .as_str()
            .unwrap_or(text(&row["status"]))
            .to_owned();
        let result = self.run_job["result"].clone();
        if self.focused != self.run {
            if let (Some(lat), Some(lon)) = (
                result["top"][0]["lat"].as_f64(),
                result["top"][0]["lon"].as_f64(),
            ) {
                self.map.focus(lat, lon); // a search opens on its best place
                self.focused = self.run.clone();
            }
        }
        egui::TopBottomPanel::top("run_head")
            .frame(egui::Frame::new().inner_margin(egui::Margin { left: 14, right: 0, top: 4, bottom: 8 }))
            .show_inside(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading(format!("Search #{number_of}"));
                    ui.label(RichText::new(format!("{} · area {}", when(&row["settings"]), area_km(&row["settings"]["box"]))).weak());
                });
                ui.horizontal_wrapped(|ui| {
                    if status == "running" || status == "queued" {
                        if ui.button("Stop…").clicked() {
                            self.confirm_stop = true;
                        }
                    }
                    if (status == "stopped" || status == "failed")
                        && ui
                            .add_enabled(!self.busy, egui::Button::new(RichText::new("Resume").strong()))
                            .on_hover_text("Carry on with the same settings and curves; the parts already searched are kept")
                            .clicked()
                    {
                        self.busy = true;
                        api::post(self.tx.clone(), "rerun".into(), format!("/api/jobs/{}/rerun", self.run), json!({}));
                    }
                    if ui
                        .button("Change settings and search again…")
                        .on_hover_text("Opens step 2 filled in with this search's settings")
                        .clicked()
                    {
                        self.fill_form(&row);
                        self.step = Step::Settings;
                    }
                    if result.is_object() && ui.button("Save result…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name(format!("doppler-{}.zip", self.run))
                            .save_file()
                        {
                            self.busy = true;
                            api::export(self.tx.clone(), self.run.clone(), path);
                        }
                    }
                });
                if let Some((title, color, detail)) = verdict(&row["verdict"]) {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(title).strong().size(16.0).color(color));
                        ui.label(RichText::new(detail).small().weak());
                    });
                } else if status == "failed" {
                    ui.colored_label(Color32::RED, format!("Failed: {}", text(&self.run_job["error"])));
                } else if status == "stopped" {
                    ui.label("Stopped before the end. Resume carries on where it stopped.");
                }
            });
        egui::SidePanel::left("candidates")
            .resizable(true)
            .default_width(330.0)
            .width_range(260.0..=520.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("run_detail").show(ui, |ui| {
                    if status == "running" || status == "queued" {
                        progress_ui(ui, &self.run_job);
                        ui.separator();
                    }
                    let top = result["top"].as_array().cloned().unwrap_or_default();
                    if !top.is_empty() {
                        ui.strong("Best places");
                        ui.label(RichText::new("Ranked by how well real flights explain your curves. Click one to see it and its planes.").small().weak());
                    }
                    for (i, c) in top.iter().take(10).enumerate() {
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .selectable_label(self.selected == i, format!("#{}  score {:.1}", i + 1, number(&c["score"])))
                                .clicked()
                            {
                                self.selected = i;
                                if let (Some(lat), Some(lon)) = (c["lat"].as_f64(), c["lon"].as_f64()) {
                                    self.map.focus(lat, lon);
                                }
                                self.track = Value::Null;
                            }
                            ui.label(RichText::new(format!("{:.5}, {:.5} · clock {:+.0} s", number(&c["lat"]), number(&c["lon"]), number(&c["clock_s"]))).small());
                        });
                        if self.selected == i {
                            if let Some(matches) = c["matches"].as_array() {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(RichText::new("Planes:").small());
                                    for m in matches {
                                        let icao = text(&m["icao"]);
                                        if ui.link(format!("{} {}", icao, m["type"].as_str().unwrap_or(""))).clicked() {
                                            self.get(&format!("track:{}", self.run), format!("/api/jobs/{}/track/{icao}", self.run));
                                        }
                                    }
                                });
                            }
                        }
                    }
                    ui.add_space(6.0);
                    egui::CollapsingHeader::new("Settings used").id_salt("settings_used").show(ui, |ui| {
                        let s = &row["settings"];
                        egui::Grid::new("settings_grid").num_columns(2).show(ui, |ui| {
                            for (k, v) in [
                                ("Recorded", when(s)),
                                ("Area", area_km(&s["box"])),
                                ("Method", method_name(text(&s["search"])).into()),
                                ("Curves", text_num(&s["curves"])),
                                ("CPU workers", text_num(&s["workers"])),
                                ("Fake sky", if s["fake_sky"] == true { "yes" } else { "no" }.into()),
                            ] {
                                ui.label(RichText::new(k).weak());
                                ui.label(v);
                                ui.end_row();
                            }
                        });
                    });
                    egui::CollapsingHeader::new("Log").id_salt("run_log").show(ui, |ui| {
                        if let Some(lines) = self.run_job["log"].as_array() {
                            for l in lines {
                                ui.label(RichText::new(text(l)).monospace().small());
                            }
                        }
                    });
                });
            });
        // the map shows this search's own area (the form's is put back after)
        let (form_box, drawing) = (self.map.bounds, self.map.drawing);
        self.map.drawing = false;
        self.map.bounds = row["settings"]["box"]
            .as_array()
            .filter(|b| b.len() == 4)
            .map(|b| [number(&b[0]), number(&b[1]), number(&b[2]), number(&b[3])]);
        if let Some(i) = self.map.ui(ui, &self.tx, &result["top"], &self.track) {
            self.selected = i;
        }
        (self.map.bounds, self.map.drawing) = (form_box, drawing);
    }
}
// The job's steps, read from the progress lines it already logs (nothing extra is written): which have started,
// and the running one's count where its lines carry one ("curve 2/5", "1 of 2").
const STEPS: [(&str, &str); 6] = [
    ("Load the audio", " audio "),
    ("Curves", " detections (score"),
    ("Flight data", "aircraft active in the box"),
    ("Ground height", "above the WGS84 ellipsoid"),
    ("Place search", "real sky:"),
    ("Fake sky", "fake sky (the flights"),
];
fn progress_ui(ui: &mut egui::Ui, job: &Value) {
    let lines: Vec<&str> = job["log"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    // the steps this job runs: curves only, or locating with or without a fake sky
    let shown = if job["params"]["detect_only"] == true {
        2
    } else if job["params"]["fake_sky"] == true {
        6
    } else {
        5
    };
    let reached = |key: &str| lines.iter().rposition(|l| l.contains(key));
    // the last step seen in the log (-1: none yet); steps 0-3 log when finished, the searches as they go
    let c = (0..STEPS.len())
        .rev()
        .find(|&i| reached(STEPS[i].1).is_some())
        .map_or(-1, |i| i as i64);
    ui.heading("Analysis running");
    for (i, (name, _)) in STEPS.iter().enumerate().take(shown) {
        let i = i as i64;
        let state = if i < c || (i == c && c < 4) {
            "done"
        } else if (i == c && c >= 4) || (i == c + 1 && c < 4) {
            "running"
        } else {
            "waiting"
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(match state {
                    "done" => "✓",
                    "running" => "●",
                    _ => "○",
                })
                .strong(),
            );
            ui.label(*name);
            if state == "running" {
                let last = lines.last().copied().unwrap_or("");
                match done_fraction(last) {
                    Some(f) => {
                        ui.add(
                            egui::ProgressBar::new(f)
                                .desired_width(220.0)
                                .text(format!("{:.0}%", f * 100.0)),
                        );
                    }
                    None => {
                        ui.spinner();
                    }
                }
            }
        });
    }
    if let Some(last) = lines.last() {
        ui.label(RichText::new(last.get(9..).unwrap_or(last)).small().weak()); // the latest line, without its time
    }
}
/// How much of the running stage is done, from its latest line: its percentage ("37%") or, failing that, its
/// last "k/n" count.
fn done_fraction(line: &str) -> Option<f32> {
    let pct = line
        .split_whitespace()
        .filter_map(|w| w.strip_suffix('%')?.parse::<f32>().ok())
        .last();
    pct.map(|p| p / 100.0)
        .or_else(|| fraction(line).map(|(k, n)| k as f32 / n as f32))
        .map(|f| f.clamp(0.0, 1.0))
}
/// The last "k/n" or "k of n" count in a progress line.
fn fraction(line: &str) -> Option<(u64, u64)> {
    let words: Vec<&str> = line
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == ',')
        .filter(|w| !w.is_empty())
        .collect();
    let mut found = None;
    for (i, w) in words.iter().enumerate() {
        if let Some((a, b)) = w.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) {
                found = Some((a, b));
            }
        } else if *w == "of" && i > 0 && i + 1 < words.len() {
            if let (Ok(a), Ok(b)) = (words[i - 1].parse::<u64>(), words[i + 1].parse::<u64>()) {
                found = Some((a, b));
            }
        }
    }
    found.filter(|&(_, n)| n > 0)
}
const ACCENT: Color32 = Color32::from_rgb(16, 105, 94);
fn method_name(id: &str) -> &'static str {
    match id {
        "tracks" => "Curve shapes vs real flight paths (best)",
        "forward" => "Track matching on the sound",
        "fits" => "Measurement fits (old)",
        _ => "?",
    }
}
/// A box [lat0, lat1, lon0, lon1] as "1 200 × 800 km" (east-west × north-south).
fn area_km(b: &Value) -> String {
    let b: Vec<f64> = b
        .as_array()
        .map_or(vec![], |b| b.iter().map(number).collect());
    if b.len() != 4 {
        return "—".into();
    }
    let ns = (b[1] - b[0]) * 111.2;
    let ew = (b[3] - b[2]) * 111.2 * ((b[0] + b[1]) / 2.0).to_radians().cos();
    format!("{ew:.0} × {ns:.0} km")
}
/// A job id's start ("20261112-094000-…") as "2026-11-12 09:40".
fn started(id: &str) -> String {
    match (
        id.get(0..4),
        id.get(4..6),
        id.get(6..8),
        id.get(9..11),
        id.get(11..13),
    ) {
        (Some(y), Some(mo), Some(d), Some(h), Some(mi)) => format!("{y}-{mo}-{d} {h}:{mi}"),
        _ => id.into(),
    }
}
/// When a search assumed the recording was made: "2026-11-11, whole day" or "2026-11-11 09:40:00 ±15 s" (UTC).
fn when(s: &Value) -> String {
    let t = text(&s["start_utc"]);
    let date = t.get(0..10).unwrap_or(t);
    if s["date_only"] == true {
        format!("{date}, whole day")
    } else {
        format!(
            "{date} {} UTC ±{} s",
            t.get(11..19).unwrap_or(""),
            text_num(&s["clock_s"])
        )
    }
}
/// A search's result in plain words, from its winner against chance (the best elsewhere or at another time) or the
/// fake sky: (title, colour, the numbers). None while it has no result.
// shortcut: the 10-point gap is a rough reading of a handful of runs, not calibrated; revisit with more searches
fn verdict(v: &Value) -> Option<(&'static str, Color32, String)> {
    if !v["best_score"].is_number() || v["detect_only"] == true {
        return None;
    }
    let (good, maybe, bad) = (
        Color32::from_rgb(25, 128, 80),
        Color32::from_rgb(176, 120, 20),
        Color32::from_rgb(170, 60, 50),
    );
    if v["fake_sky_best"].is_number() {
        let detail = format!(
            "best {:.1} vs fake sky {:.1}",
            number(&v["best_score"]),
            number(&v["fake_sky_best"])
        );
        return Some(if v["beats_fake_sky"] == true {
            ("Match found: it stands out from chance", good, detail)
        } else {
            (
                "No clear match: the best place scores like chance",
                bad,
                detail,
            )
        });
    }
    if let Some(c) = v["chance_elsewhere"].as_object() {
        let (w, ch) = (number(&c["winner_same_stage"]), number(&c["chance_score"]));
        let detail = format!("winner {w:.1} vs chance {ch:.1} (best elsewhere or at another time)");
        return Some(if w - ch >= 10.0 {
            ("Clear match", good, detail)
        } else if w > ch {
            ("Possible match: check it", maybe, detail)
        } else {
            (
                "No clear match: the places below are no better than chance",
                bad,
                detail,
            )
        });
    }
    Some((
        "Not compared with chance",
        Color32::from_rgb(110, 120, 118),
        format!(
            "best score {:.1}; tick the fake sky to measure chance",
            number(&v["best_score"])
        ),
    ))
}
/// A search's state as a small coloured label (its verdict once done).
fn status_chip(ui: &mut egui::Ui, r: &Value) {
    let (label, color) = match text(&r["status"]) {
        "running" => ("running", Color32::from_rgb(30, 100, 170)),
        "queued" => ("queued", Color32::from_rgb(110, 120, 118)),
        "stopped" => ("stopped", Color32::from_rgb(176, 120, 20)),
        "failed" => ("failed", Color32::from_rgb(170, 60, 50)),
        _ => match verdict(&r["verdict"]) {
            Some((t, c, _)) if t.starts_with("Clear") || t.starts_with("Match") => ("match", c),
            Some((t, c, _)) if t.starts_with("Possible") => ("possible", c),
            Some((t, c, _)) if t.starts_with("No clear") => ("no match", c),
            _ => ("done", Color32::from_rgb(110, 120, 118)),
        },
    };
    ui.label(RichText::new(label).small().strong().color(color));
}
fn text_num(v: &Value) -> String {
    if v.is_string() {
        text(v).to_owned()
    } else if v.is_number() {
        v.to_string()
    } else {
        "—".into()
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.events(ctx);
        ctx.request_repaint_after(Duration::from_millis(100));
        if self.last_jobs.elapsed() > Duration::from_secs(10) {
            self.get("jobs", "/api/jobs".into());
            self.last_jobs = Instant::now();
        }
        if !self.current.is_empty() && self.last_job.elapsed() > Duration::from_secs(2) {
            self.get(
                &format!("job:{}", self.current),
                format!("/api/jobs/{}", self.current),
            );
            self.last_job = Instant::now();
        }
        if !self.run.is_empty() && self.last_run.elapsed() > Duration::from_secs(2) {
            self.get(
                &format!("run:{}", self.run),
                format!("/api/jobs/{}", self.run),
            );
            self.last_run = Instant::now();
        }
        if self.render_due.is_some_and(|t| Instant::now() >= t)
            && self.render_inflight.is_none()
            && self.job["has_audio"] == true
        {
            self.render_due = None;
            let span = self.t1 - self.t0;
            let pps = (900.0 / span).clamp(0.02, 8.0);
            let view = Piece {
                t0: self.t0,
                t1: self.t1,
                fmin: self.fmin,
                fmax: self.fmax,
                log: self.log_axis,
                win: self.win,
                pps,
            };
            if self.pieces.iter().any(|q| q.w.covers(&view)) {
                self.rendering = false; // seen already, sharp enough: nothing to fetch
            } else {
                // half a view more on each side, so a short swipe shows a ready image
                let duration = number(&self.summary["duration_s"]).max(self.t1);
                let t0 = (self.t0 - span / 2.0).max(0.0);
                let t1 = (self.t1 + span / 2.0).min(duration).min(t0 + 3600.0);
                let w = Piece { t0, t1, ..view };
                self.render_inflight = Some(self.render_seq);
                self.requested = Some(w);
                api::image(
                    self.tx.clone(),
                    format!("render:{}:{}", self.current, self.render_seq),
                    format!(
                        "/api/jobs/{}/render?t0={}&t1={}&fmin={}&fmax={}&axis={}&win={}&pps={}&h=500",
                        self.current,
                        w.t0,
                        w.t1,
                        w.fmin,
                        w.fmax,
                        if w.log { "log" } else { "lin" },
                        w.win,
                        pps
                    ),
                    false,
                );
            }
        }
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(25, 50, 47))
                    .inner_margin(egui::Margin::symmetric(20, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("◉  Dopplero")
                            .size(17.0)
                            .strong()
                            .color(Color32::from_rgb(242, 249, 245)),
                    );
                    ui.label(
                        RichText::new(
                            "find where a recording was made from the planes heard in it",
                        )
                        .size(12.0)
                        .color(Color32::from_rgb(169, 200, 190)),
                    );
                });
            });
        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(280.0)
            .width_range(240.0..=420.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(14, 14))
                    .stroke(egui::Stroke::new(1.0_f32, Color32::from_rgb(226, 232, 228))),
            )
            .show(ctx, |ui| {
                egui::TopBottomPanel::bottom("sidebar_foot")
                    .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 8)))
                    .show_inside(ui, |ui| {
                        if ui
                            .button("Import a saved result…")
                            .on_hover_text("A .zip made with “Save result…” (it does not include the recording)")
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new().add_filter("Saved result", &["zip"]).pick_file() {
                                self.busy = true;
                                api::import(self.tx.clone(), path);
                            }
                        }
                    });
                if ui
                    .add(
                        egui::Button::new(RichText::new("+  New recording").strong().color(Color32::WHITE))
                            .fill(ACCENT)
                            .min_size(Vec2::new(ui.available_width(), 38.0)),
                    )
                    .clicked()
                {
                    self.new_open = true;
                }
                ui.add_space(10.0);
                ui.label(RichText::new("RECORDINGS").small().strong().color(Color32::from_rgb(110, 125, 120)));
                let jobs = self.jobs.as_array().cloned().unwrap_or_default();
                // one entry per recording: the jobs whose root is themselves
                let roots: Vec<&Value> = jobs.iter().filter(|j| j["root"] == j["id"]).collect();
                if roots.is_empty() {
                    ui.label(RichText::new("None yet. Start with “New recording”.").small().weak());
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for j in roots {
                        let id = text(&j["id"]);
                        let runs = self.searches(id);
                        let sub = match (text(&j["status"]), runs.first()) {
                            ("queued" | "running", _) if j["detect_only"] == true => "loading…".to_string(),
                            ("failed", _) if j["detect_only"] == true => "could not be loaded".to_string(),
                            (_, None) => "curves only, no search yet".to_string(),
                            (_, Some(last)) => format!(
                                "{} search{} · last: {}",
                                runs.len(),
                                if runs.len() == 1 { "" } else { "es" },
                                match text(&last["status"]) {
                                    "done" => verdict(&last["verdict"]).map_or("done", |(t, ..)| {
                                        if t.starts_with("Clear") || t.starts_with("Match") {
                                            "match"
                                        } else if t.starts_with("Possible") {
                                            "possible match"
                                        } else if t.starts_with("No clear") {
                                            "no match"
                                        } else {
                                            "done"
                                        }
                                    }),
                                    s => s,
                                }
                            ),
                        };
                        let on = self.current == id;
                        let resp = egui::Frame::new()
                            .fill(if on { Color32::from_rgb(226, 242, 236) } else { Color32::from_rgb(248, 250, 248) })
                            .corner_radius(8)
                            .inner_margin(egui::Margin::symmetric(10, 8))
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.label(RichText::new(text(&j["file"])).strong());
                                ui.label(RichText::new(format!("{} · {sub}", started(id))).small().weak());
                            })
                            .response
                            .interact(egui::Sense::click());
                        if resp.clicked() && !on {
                            self.open(id.into());
                        }
                        ui.add_space(2.0);
                    }
                });
            });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(246, 247, 245))
                    .inner_margin(egui::Margin::symmetric(20, 12)),
            )
            .show(ctx, |ui| {
                if self.current.is_empty() {
                    ui.add_space(60.0);
                    ui.vertical_centered(|ui| {
                        ui.heading("Where was it recorded?");
                        ui.label("1. Open a recording   2. Mark the planes' curves on its spectrogram   3. Say when and roughly where   4. Get the likeliest places");
                        ui.add_space(12.0);
                        if ui
                            .add(egui::Button::new(RichText::new("+  New recording").strong().color(Color32::WHITE)).fill(ACCENT))
                            .clicked()
                        {
                            self.new_open = true;
                        }
                        ui.label(RichText::new("Or pick one on the left to carry on with it.").small().weak());
                        if !self.status.is_empty() {
                            ui.label(RichText::new(&self.status).color(Color32::from_rgb(156, 82, 39)));
                        }
                    });
                } else {
                    let title = self
                        .jobs
                        .as_array()
                        .and_then(|a| a.iter().find(|j| j["id"] == self.current.as_str()))
                        .map_or("Recording", |j| text(&j["file"]))
                        .to_owned();
                    ui.heading(title);
                    ui.add_space(4.0);
                    let n = self.searches(&self.current).len();
                    ui.horizontal(|ui| {
                        for (step, label) in [
                            (Step::Curves, "①  Mark curves".to_string()),
                            (Step::Settings, "②  Search settings".to_string()),
                            (Step::Results, format!("③  Results ({n})")),
                        ] {
                            let on = self.step == step;
                            if ui
                                .add(
                                    egui::Button::new(RichText::new(label).size(15.0).strong().color(if on { Color32::WHITE } else { Color32::from_rgb(35, 52, 50) }))
                                        .fill(if on { ACCENT } else { Color32::WHITE })
                                        .min_size(Vec2::new(170.0, 34.0)),
                                )
                                .clicked()
                            {
                                self.step = step;
                            }
                        }
                        if !self.status.is_empty() {
                            ui.label(RichText::new(&self.status).color(Color32::from_rgb(156, 82, 39)));
                        }
                    });
                    ui.add_space(6.0);
                    ui.separator();
                    match self.step {
                        Step::Curves => self.curves_step(ui),
                        Step::Settings => self.settings_step(ui),
                        Step::Results => self.results_step(ui),
                    }
                }
            });
        if self.new_open {
            self.new_recording_ui(ctx);
        }
        if self.confirm_stop {
            egui::Window::new("Stop this search?")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("You can resume it later: the parts already searched are kept.");
                    ui.horizontal(|ui| {
                        if ui.button("Keep running").clicked() {
                            self.confirm_stop = false;
                        }
                        if ui.button("Stop").clicked() {
                            self.confirm_stop = false;
                            self.busy = true;
                            api::post(
                                self.tx.clone(),
                                "stop".into(),
                                format!("/api/jobs/{}/stop", self.run),
                                json!({}),
                            );
                        }
                    });
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::zoom_interval;
    #[test]
    fn a_seen_image_covers_a_view_inside_it_but_not_a_sharper_or_wider_one() {
        let seen = super::Piece {
            t0: 100.0,
            t1: 400.0,
            fmin: 40.0,
            fmax: 800.0,
            log: false,
            win: 1.0,
            pps: 3.0,
        };
        assert!(seen.covers(&super::Piece {
            t0: 150.0,
            t1: 350.0,
            pps: 4.0,
            ..seen
        }));
        assert!(!seen.covers(&super::Piece {
            t0: 150.0,
            t1: 450.0,
            ..seen
        }));
        assert!(!seen.covers(&super::Piece {
            t0: 200.0,
            t1: 210.0,
            pps: 8.0,
            ..seen
        }));
        assert!(!seen.covers(&super::Piece {
            fmax: 1000.0,
            ..seen
        }));
        assert!(!seen.covers(&super::Piece { win: 4.0, ..seen }));
    }
    #[test]
    fn progress_counts_are_read_from_log_lines() {
        assert_eq!(
            super::fraction("20:05:35 real sky: coarse grid: curve 2/5"),
            Some((2, 5))
        );
        assert_eq!(
            super::fraction("0 s  proposing from curve 4 (1 of 2 anchor curves), 3077 flights"),
            Some((1, 2))
        );
        assert_eq!(super::fraction("ground 20 m above sea level"), None);
        assert_eq!(
            super::done_fraction("00:12:30 real sky: proposing places: 37% (anchor curve 1 of 2)"),
            Some(0.37)
        );
        assert_eq!(super::done_fraction("real sky: proposing 3/4"), Some(0.75));
    }
    #[test]
    fn start_time_is_iso_utc_and_months_have_their_days() {
        assert_eq!(
            super::start_utc([2026, 9, 8, 20, 17, 29]),
            "2026-09-08T20:17:29Z"
        );
        assert_eq!(
            (
                super::days_in_month(2024, 2),
                super::days_in_month(1900, 2),
                super::days_in_month(2026, 4)
            ),
            (29, 28, 30)
        );
        let [y, m, d, ..] = super::today_noon();
        assert!(
            y >= 2024 && (1..=12).contains(&m) && (1..=super::days_in_month(y, m)).contains(&d)
        );
    }
    #[test]
    fn zoom_keeps_anchor_in_place_and_respects_bounds() {
        let (lo, hi) = zoom_interval(20.0, 80.0, 35.0, 0.5, 0.0, 100.0, 1.0);
        assert!((lo - 27.5).abs() < 1e-9 && (hi - 57.5).abs() < 1e-9);
        assert_eq!(
            zoom_interval(0.0, 30.0, 0.0, 10.0, 0.0, 100.0, 1.0),
            (0.0, 100.0)
        );
        assert_eq!(
            zoom_interval(0.0, 0.5, 0.25, 0.1, 0.0, 0.5, 1.0),
            (0.0, 0.5)
        );
    }
}
