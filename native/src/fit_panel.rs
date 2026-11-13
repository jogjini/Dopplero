//! The curve tool on the Recording page: start a curve (from scratch or from a detected one), drag it onto a
//! line, click points on the line (it re-fits after each), snap it to the bright line, fine-tune with sliders,
//! save. The maths is in curve.rs.
use crate::curve::{trace_box, Curve, Point, Window, MODELS};
use eframe::egui::{self, Color32, Pos2, RichText};
use serde_json::Value;

/// Where the spectrogram is drawn and what it shows: converts (time, frequency) to screen and back.
#[derive(Clone, Copy)]
pub struct View {
    pub rect: egui::Rect,
    pub t0: f64,
    pub t1: f64,
    pub fmin: f64,
    pub fmax: f64,
    pub log: bool,
}
impl View {
    pub fn pos(&self, t: f64, f: f64) -> Option<Pos2> {
        if !(t.is_finite() && f.is_finite()) || (self.log && f <= 0.0) {
            return None;
        }
        let q = if self.log {
            (f / self.fmin).ln() / (self.fmax / self.fmin).ln()
        } else {
            (f - self.fmin) / (self.fmax - self.fmin)
        };
        Some(egui::pos2(
            self.rect.left()
                + (self.rect.width() as f64 * (t - self.t0) / (self.t1 - self.t0)) as f32,
            self.rect.bottom() - (self.rect.height() as f64 * q) as f32,
        ))
    }
    pub fn at(&self, p: Pos2) -> (f64, f64) {
        let x = ((p.x - self.rect.left()) / self.rect.width()) as f64;
        let q = ((self.rect.bottom() - p.y) / self.rect.height()) as f64;
        let f = if self.log {
            self.fmin * (self.fmax / self.fmin).powf(q)
        } else {
            self.fmin + q * (self.fmax - self.fmin)
        };
        (self.t0 + x * (self.t1 - self.t0), f)
    }
    pub fn window(&self) -> Window {
        Window {
            t0: self.t0,
            t1: self.t1,
            fmin: self.fmin,
            fmax: self.fmax,
        }
    }
}

const TONES: [f64; 8] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];
const ACTIVE: Color32 = Color32::from_rgb(255, 176, 46);
const SAVED: Color32 = Color32::from_rgb(236, 110, 220);

fn tone_label(h: f64) -> String {
    if h == 0.5 {
        "×½".into()
    } else {
        format!("×{h}")
    }
}

type Box2 = ((f64, f64), (f64, f64));

/// Box a line: a drag draws a box, and the brightest continuous line in it becomes points.
pub struct BoxTrace {
    pub on: bool,
    /// Where the drag started (time, frequency), while a box is being drawn.
    pub from: Option<(f64, f64)>,
    /// A box just drawn, to trace: (time span, frequency span).
    pub drawn: Option<Box2>,
    /// The last box traced and the points it added (the sliders re-trace it).
    last: Option<(Box2, Vec<Point>)>,
    smooth: f64,
    sensitivity: f32,
}
impl Default for BoxTrace {
    fn default() -> Self {
        BoxTrace {
            on: false,
            from: None,
            drawn: None,
            last: None,
            smooth: 1.0,
            sensitivity: 0.15,
        }
    }
}

#[derive(Default)]
pub struct FitTool {
    /// The curve being edited, and which saved curve it replaces (None = a new one).
    pub active: Option<Curve>,
    pub editing: Option<usize>,
    /// The job's hand-fitted curves (manual.json).
    pub saved: Vec<Curve>,
    /// The tone new points go on.
    pub tone: Option<f64>,
    pub new_model: usize,
    /// Plain clicks on the spectrogram add points (off: clicks select detected curves).
    pub click_adds: bool,
    /// The curve is being dragged (instead of the view).
    pub dragging: bool,
    /// The curve isn't drawn (points first: the curve appears when fitted to them).
    pub hidden: bool,
    /// Re-fit after each click (off: the curve stays put while points are placed; Re-fit when ready).
    pub auto_fit: bool,
    pub boxer: BoxTrace,
    /// The mouse erases: a click removes the nearest point, a dragged box every point in it.
    pub erasing: bool,
    pub note: String,
}

impl FitTool {
    pub fn load(&mut self, v: &Value) {
        self.saved = v
            .as_array()
            .map(|a| a.iter().filter_map(Curve::from_json).collect())
            .unwrap_or_default();
    }
    /// The saved curves as the service stores them.
    pub fn to_json(&self) -> Value {
        Value::Array(self.saved.iter().map(Curve::to_json).collect())
    }
    fn begin(&mut self, c: Curve, editing: Option<usize>) {
        self.active = Some(c);
        self.editing = editing;
        self.tone = Some(1.0);
        self.click_adds = true;
        self.hidden = false;
        self.boxer.on = false;
        self.boxer.last = None;
        self.erasing = false;
        self.note.clear();
    }

    /// Draw the saved curves and the one being edited (stronger over its saved span, faint beyond).
    pub fn draw(&self, p: &egui::Painter, v: &View) {
        for (i, c) in self.saved.iter().enumerate() {
            if self.editing != Some(i) || self.active.is_none() {
                for h in c.tones() {
                    line(
                        p,
                        v,
                        c,
                        h,
                        c.extent,
                        egui::Stroke::new(2.0_f32, SAVED),
                        false,
                    );
                }
            }
        }
        let Some(c) = &self.active else { return };
        if let Some((((t0, t1), (f0, f1)), _)) = self.boxer.last {
            if let (Some(a), Some(b)) = (v.pos(t0, f0), v.pos(t1, f1)) {
                let r = egui::Rect::from_two_pos(a, b);
                let s =
                    egui::Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 110));
                p.extend(egui::Shape::dashed_line(
                    &[
                        r.left_top(),
                        r.right_top(),
                        r.right_bottom(),
                        r.left_bottom(),
                        r.left_top(),
                    ],
                    s,
                    5.0,
                    4.0,
                ));
            }
        }
        let anchor = c.data().iter().map(|q| q.tone).fold(1.0_f64, f64::min);
        if self.hidden {
            for q in &c.points {
                if let Some(pos) = v.pos(q.t, q.f).filter(|pos| v.rect.contains(*pos)) {
                    p.circle_stroke(pos, 4.0, egui::Stroke::new(2.0_f32, Color32::WHITE));
                }
            }
            return;
        }
        for q in &c.traced {
            if let Some(pos) = v.pos(q.t, q.f).filter(|pos| v.rect.contains(*pos)) {
                p.circle_filled(pos, 1.5, Color32::from_rgb(120, 220, 255));
            }
        }
        for h in c.tones() {
            let main = h == anchor;
            let faint = Color32::from_rgba_unmultiplied(255, 176, 46, 90);
            line(
                p,
                v,
                c,
                h,
                (v.t0, v.t1),
                egui::Stroke::new(1.0_f32, faint),
                !main,
            );
            line(
                p,
                v,
                c,
                h,
                c.extent,
                egui::Stroke::new(if main { 2.5_f32 } else { 1.5 }, ACTIVE),
                !main,
            );
        }
        for q in &c.points {
            if let Some(pos) = v.pos(q.t, q.f).filter(|pos| v.rect.contains(*pos)) {
                p.circle_stroke(pos, 4.0, egui::Stroke::new(2.0_f32, Color32::WHITE));
            }
        }
    }

    /// A click on the spectrogram while a curve is edited adds a point to the chosen tone (plain click when
    /// "click adds points" is on, Ctrl/Cmd-click always; Alt-click removes the nearest) and re-fits.
    /// Returns true when the click was the tool's.
    pub fn click(&mut self, v: &View, pos: Pos2, mods: egui::Modifiers) -> bool {
        if self.active.is_none() {
            return false;
        }
        if mods.alt || self.erasing {
            self.remove_near(v, pos);
            return true;
        }
        if !(mods.command || self.click_adds) {
            return false;
        }
        let tone = self.tone.unwrap_or(1.0);
        let c = self.active.as_mut().unwrap();
        let (t, f) = v.at(pos);
        c.points.push(Point { t, f, tone });
        self.points_changed();
        true
    }
    /// After points were added or removed: the saved line reaches every point, and the curve re-fits (when it should).
    fn points_changed(&mut self) {
        let Some(c) = &mut self.active else { return };
        if c.points.len() >= 2 {
            for t in c.points.iter().map(|q| q.t) {
                c.extent = (c.extent.0.min(t), c.extent.1.max(t));
            }
        }
        if self.auto_fit && !self.hidden {
            refit(c);
        } else {
            c.refresh_offsets();
        }
    }
    /// Remove every point (clicked, boxed or traced) inside a box.
    pub fn erase(&mut self, ((t0, t1), (f0, f1)): Box2) {
        let Some(c) = &mut self.active else { return };
        let inside = |q: &Point| {
            q.t >= t0.min(t1) && q.t <= t0.max(t1) && q.f >= f0.min(f1) && q.f <= f0.max(f1)
        };
        let n = c.points.len() + c.traced.len();
        c.points.retain(|q| !inside(q));
        c.traced.retain(|q| !inside(q));
        if self
            .boxer
            .last
            .as_ref()
            .is_some_and(|(_, added)| added.iter().any(inside))
        {
            self.boxer.last = None; // its sliders would bring the erased points back
        }
        self.note = format!("Erased {} points.", n - c.points.len() - c.traced.len());
        self.points_changed();
    }
    /// Fill a box with the points of its brightest line (on the chosen tone), replacing the last box's points when
    /// it is re-traced.
    fn trace(&mut self, bx: Box2, bright: &dyn Fn(f64, f64) -> Option<f32>) {
        let Some(c) = &mut self.active else { return };
        if let Some((_, old)) = self.boxer.last.take() {
            c.points.retain(|q| !old.contains(q));
        }
        let tone = self.tone.unwrap_or(1.0);
        let new: Vec<Point> = trace_box(
            bright,
            bx.0,
            bx.1,
            self.boxer.smooth,
            self.boxer.sensitivity,
        )
        .into_iter()
        .map(|(t, f)| Point { t, f, tone })
        .collect();
        self.note = match new.len() {
            0 => "No line stands out in the box: lower “stands out by”, or draw it where the spectrogram is loaded.".into(),
            n => format!("The box added {n} points."),
        };
        c.points.extend(&new);
        self.boxer.last = Some((bx, new));
        self.points_changed();
    }
    /// The box tool's controls; traces a box just drawn.
    fn box_ui(&mut self, ui: &mut egui::Ui, bright: &dyn Fn(f64, f64) -> Option<f32>) {
        if let Some(bx) = self.boxer.drawn.take() {
            self.trace(bx, bright);
        }
        ui.horizontal(|ui| {
            ui.label("Mouse:");
            ui.selectable_value(&mut self.erasing, false, "Add points");
            ui.selectable_value(&mut self.erasing, true, "Erase points")
                .on_hover_text(
                    "A click removes the nearest point; drag a box to remove every point in it",
                );
        });
        if !self.erasing && (self.boxer.on || self.boxer.last.is_some()) {
            ui.checkbox(
                &mut self.boxer.on,
                "Drags draw boxes that fill with points (off: drags scroll and move the curve)",
            );
        }
        let Some((bx, added)) = &self.boxer.last else {
            return;
        };
        let (bx, n) = (*bx, added.len());
        let mut again = false;
        again |= ui
            .add(egui::Slider::new(&mut self.boxer.sensitivity, 0.0..=0.5).text("stands out by"))
            .on_hover_text(
                "Lower: more points, also where the line is faint. Higher: only where it is clear",
            )
            .changed();
        again |= ui
            .add(egui::Slider::new(&mut self.boxer.smooth, 0.0..=10.0).text("smoothness"))
            .on_hover_text(
                "Higher: the line can't jump to another line, but may cut a steep fall short",
            )
            .changed();
        if again {
            self.trace(bx, bright);
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button(format!("Undo last box ({n} points)")).clicked() {
                if let (Some(c), Some((_, old))) = (&mut self.active, self.boxer.last.take()) {
                    c.points.retain(|q| !old.contains(q));
                }
                self.points_changed();
            }
        });
        ui.label(RichText::new("Wrong line? Undo and draw a tighter box, or raise smoothness. The sliders adjust the last box; a new box adds its points. Right-click removes single points.").small().weak());
    }
    /// Remove the user's point nearest to pos (within 12 px) and re-fit.
    pub fn remove_near(&mut self, v: &View, pos: Pos2) {
        let Some(c) = &mut self.active else { return };
        let near = c
            .points
            .iter()
            .enumerate()
            .filter_map(|(i, q)| Some((i, v.pos(q.t, q.f)?.distance(pos))))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = near {
            if d <= 12.0 {
                c.points.remove(i);
                self.points_changed();
            }
        }
    }
    /// Whether pos is on (within 10 px of) a tone of the curve being edited: a drag there moves the curve.
    pub fn near_curve(&self, v: &View, pos: Pos2) -> bool {
        let Some(c) = self.active.as_ref().filter(|_| !self.hidden) else {
            return false;
        };
        let (t, _) = v.at(pos);
        c.tones().into_iter().any(|h| {
            v.pos(t, c.eval(t, h))
                .is_some_and(|q| (q.y - pos.y).abs() <= 10.0)
        })
    }
    /// Move the curve with the pointer, from one screen position to the next.
    pub fn drag(&mut self, v: &View, from: Pos2, to: Pos2) {
        let Some(c) = &mut self.active else { return };
        let ((t1, f1), (t2, f2)) = (v.at(from), v.at(to));
        if f1 > 0.0 && f2 > 0.0 {
            c.shift(t2 - t1, f2 / f1);
            c.refresh_offsets();
        }
    }

    /// The panel with no curve open: start a curve, or edit / delete a saved one. True when the saved curves
    /// changed (to send to the service).
    pub fn start_ui(&mut self, ui: &mut egui::Ui, view: &View) -> bool {
        let mut changed = false;
        ui.heading("Mark a curve");
        ui.label(
            RichText::new("Trace each plane's line (its pitch falling as it passes). Three ways:")
                .small()
                .weak(),
        );
        let wide = egui::vec2(ui.available_width(), 0.0);
        if ui.add(egui::Button::new(RichText::new("Box a line").strong()).min_size(wide)).on_hover_text("Drag a box around a line: it fills with points along the line; then fit the curve to them").clicked() {
            self.begin(Curve::new(self.new_model, vec![], view.window()), None);
            self.hidden = true;
            self.boxer.on = true;
        }
        if ui
            .add(egui::Button::new("Click points along a line").min_size(wide))
            .on_hover_text("Click points along a line; then fit the curve to them")
            .clicked()
        {
            self.begin(Curve::new(self.new_model, vec![], view.window()), None);
            self.hidden = true;
        }
        if ui
            .add(egui::Button::new("Start from a curve in this view").min_size(wide))
            .on_hover_text(
                "A curve appears in the middle of the view: drag it onto the line and snap it",
            )
            .clicked()
        {
            self.begin(Curve::new(self.new_model, vec![], view.window()), None);
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Shape").small().weak());
            egui::ComboBox::from_id_salt("new_model")
                .selected_text(MODELS[self.new_model].label())
                .show_ui(ui, |ui| {
                    for (i, m) in MODELS.iter().enumerate() {
                        ui.selectable_value(&mut self.new_model, i, m.label());
                    }
                });
        });
        ui.add_space(8.0);
        ui.separator();
        ui.strong(format!("Your curves ({})", self.saved.len()));
        if self.saved.is_empty() {
            ui.label(
                RichText::new(
                    "None yet. Curves are saved as you go: you can close the app and come back.",
                )
                .small()
                .weak(),
            );
        }
        let mut edit = None;
        let mut delete = None;
        for (i, c) in self.saved.iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("●").color(SAVED));
                ui.label(summary(c));
                if ui.small_button("Edit").clicked() {
                    edit = Some(i);
                }
                if ui.small_button("Delete").clicked() {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = edit {
            self.begin(self.saved[i].clone(), Some(i));
        }
        if let Some(i) = delete {
            self.saved.remove(i);
            changed = true;
        }
        changed
    }

    /// The open curve's title and Save / Cancel, kept in view above its controls. True when the saved curves
    /// changed.
    pub fn header_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.heading(match (self.editing.is_some(), self.hidden) {
                (true, _) => "Edit curve",
                (false, true) => "New curve: points",
                (false, false) => "New curve",
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.hidden
                    && ui
                        .add(
                            egui::Button::new(
                                RichText::new("Save curve").strong().color(Color32::WHITE),
                            )
                            .fill(Color32::from_rgb(16, 105, 94)),
                        )
                        .clicked()
                {
                    let done = self.active.take().unwrap();
                    match self.editing.take() {
                        Some(i) => self.saved[i] = done,
                        None => self.saved.push(done),
                    }
                    self.dragging = false;
                    changed = true;
                } else if ui.button("Cancel").clicked() {
                    self.active = None;
                    self.editing = None;
                    self.dragging = false;
                }
            });
        });
        changed
    }

    /// The controls of the curve being edited (shown beside the spectrogram, under header_ui). `bright(t, f)`: the
    /// loaded spectrogram's brightness (0-1), for snapping.
    pub fn edit_ui(
        &mut self,
        ui: &mut egui::Ui,
        view: &View,
        bright: &dyn Fn(f64, f64) -> Option<f32>,
    ) {
        if self.active.is_none() {
            return;
        }
        self.box_ui(ui, bright);
        ui.add_space(4.0);
        let Some(c) = &mut self.active else {
            return;
        };
        if self.hidden {
            ui.label(if self.boxer.on {
                "Drag a box around the line: it fills with points (clicks add single ones, right-click removes one). The curve appears when you fit it to them."
            } else {
                "Click points along the line (right-click removes one). The curve appears when you fit it to them."
            });
            ui.label(RichText::new("Spread them over the whole line, both sides of the steep fall. Zoom and scroll as you like meanwhile.").small().weak());
            let n = c.points.len();
            if ui
                .add_enabled(
                    n >= 4,
                    egui::Button::new(RichText::new("Fit curve to the points").strong()),
                )
                .clicked()
            {
                c.from_points(view.window());
                self.hidden = false;
                self.note = format!(
                    "Fitted to {n} points. Snap to the line refines it on the spectrogram."
                );
            }
            ui.label(
                RichText::new(format!(
                    "{n} point(s){}",
                    if n < 4 { ": at least 4 needed" } else { "" }
                ))
                .small(),
            );
            return;
        }
        ui.label(
            RichText::new("1. Drag the orange curve onto the line, or click points along it and press Re-fit.\n2. Snap to the line.\n3. Still off at one end? Free “Turn rate” or “Speeding up” below and snap again.")
                .small(),
        );
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button(RichText::new("Snap to the line").strong()).on_hover_text("Finds the bright line near the curve over its span (only where the spectrogram is loaded), then fits it. Repeat after moving the curve.").clicked() {
                let n = c.snap(bright);
                self.note = if n == 0 {
                    "No line stands out near the curve: drag it closer, or show the whole pass first.".into()
                } else {
                    format!("Snapped to {n} points of the line.")
                };
            }
            if ui.add_enabled(!c.data().is_empty(), egui::Button::new("Re-fit")).clicked() {
                c.fit();
            }
            if ui
                .add_enabled(c.data().len() >= 6, egui::Button::new("Find what bends it"))
                .on_hover_text("Frees turn rate, speeding up, climb, height or engine drift, one at a time, the one that fits the points best each time, while it helps")
                .clicked()
            {
                self.note = match c.free_what_bends_it() {
                    Some((label, a, b)) => format!("Freed “{label}”: {a:.0} → {b:.0} cents off."),
                    None => "Freeing one more setting doesn't fit the points much better.".into(),
                };
            }
            if ui.add_enabled(!c.data().is_empty(), egui::Button::new("Clear points")).clicked() {
                c.points.clear();
                c.traced.clear();
                c.refresh_offsets();
            }
        });
        ui.label(RichText::new("Right-click or Alt-click a point to remove it · drag the curve to move it · drag elsewhere to scroll").small().weak());
        if !self.note.is_empty() {
            ui.label(RichText::new(&self.note).small());
        }
        ui.label(match c.rms_cents() {
            Some(r) => format!(
                "{} clicked + {} traced points · {:.1} cents off (RMS)",
                c.points.len(),
                c.traced.len(),
                r
            ),
            None => "No points yet".into(),
        });
        if c.rms_cents().is_some_and(|r| r > 15.0) {
            ui.label(RichText::new("Still well off the points: the plane may not fly straight and steady. Try “Find what bends it”.").small().color(Color32::from_rgb(200, 120, 40)));
        }
        if let Some(pass) = c.model().pass(&c.params) {
            ui.label(
                RichText::new(format!(
                    "Closest at {}  ·  {:.0} m/s ({:.0} km/h)  ·  {:.2} km away",
                    crate::time(pass.t_cpa_s),
                    pass.speed_mps,
                    pass.speed_mps * 3.6,
                    pass.distance_m / 1000.0
                ))
                .strong(),
            );
        }
        ui.separator();
        egui::CollapsingHeader::new("Clicking")
            .id_salt("fit_clicking")
            .show(ui, |ui| {
                ui.checkbox(&mut self.click_adds, "Clicks on the spectrogram add points");
                ui.checkbox(
                    &mut self.auto_fit,
                    "Re-fit after each click (off: press Re-fit when ready)",
                );
                ui.checkbox(&mut self.hidden, "Hide the curve while placing points");
            });
        let Some(c) = &mut self.active else {
            return;
        };
        egui::CollapsingHeader::new("Fine-tune the shape")
            .id_salt("fit_tune")
            .show(ui, |ui| {
                let mut model = c.model;
                egui::ComboBox::from_id_salt("fit_model")
                    .selected_text(c.model().label())
                    .show_ui(ui, |ui| {
                        for (i, m) in MODELS.iter().enumerate() {
                            ui.selectable_value(&mut model, i, m.label());
                        }
                    });
                if model != c.model {
                    c.set_model(model, view.window());
                }
                ui.label(
                    RichText::new("Fine-tune · “free” settings are fitted, the others stay as set")
                        .small()
                        .weak(),
                );
                let mut moved = false;
                for (i, p) in c.model().params().iter().enumerate() {
                    ui.horizontal(|ui| {
                        let mut free = !c.locked[i];
                        if ui
                            .checkbox(&mut free, "")
                            .on_hover_text("free: fitted to the points")
                            .changed()
                        {
                            c.locked[i] = !free;
                        }
                        ui.label(p.label);
                    });
                    let (lo, hi) = if p.min.is_finite() && p.max.is_finite() {
                        (p.min, p.max)
                    } else {
                        (view.t0, view.t1)
                    };
                    let slider = egui::Slider::new(&mut c.params[i], lo..=hi)
                        .logarithmic(p.log)
                        .clamping(egui::SliderClamping::Never)
                        .suffix(format!(" {}", p.unit))
                        .max_decimals(if p.log { 2 } else { 1 });
                    moved |= ui.add(slider).changed();
                }
                if moved {
                    c.clamp();
                    c.refresh_offsets();
                }
            });
        egui::CollapsingHeader::new("Tones and span").id_salt("fit_tones").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Show tones");
            for h in TONES {
                let mut on = c.harmonics.contains(&h);
                if ui.checkbox(&mut on, tone_label(h)).changed() {
                    if on {
                        c.harmonics.push(h);
                    } else {
                        c.harmonics.retain(|x| *x != h);
                    }
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Clicks go on tone");
            egui::ComboBox::from_id_salt("point_tone")
                .selected_text(tone_label(self.tone.unwrap_or(1.0)))
                .show_ui(ui, |ui| {
                    for h in TONES {
                        ui.selectable_value(&mut self.tone, Some(h), tone_label(h));
                    }
                });
        });
        if ui.checkbox(&mut c.free_tones, "Each tone keeps its own pitch").on_hover_text("Real aircraft tones are often not exact multiples (propeller vs engine): leave on unless you know they are").changed() {
            c.refresh_offsets();
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Heard from");
            ui.add(
                egui::DragValue::new(&mut c.extent.0)
                    .speed(0.5)
                    .suffix(" s"),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut c.extent.1)
                    .speed(0.5)
                    .suffix(" s"),
            );
            if ui
                .add_enabled(c.data().len() > 1, egui::Button::new("Span of the points"))
                .clicked()
            {
                c.extent_from_points();
            }
            if c.extent.1 <= c.extent.0 {
                c.extent.1 = c.extent.0 + 1.0;
            }
        });
        ui.label(RichText::new("Snap and the saved line cover this span: keep it to where the line is really heard.").small().weak());
        });
    }
}

fn refit(c: &mut Curve) {
    if c.data().len() >= 3 {
        c.fit();
    } else {
        c.refresh_offsets();
    }
}

fn summary(c: &Curve) -> String {
    let m = c.model();
    let what = match m.pass(&c.params) {
        Some(p) => format!(
            "pass at {} · {:.0} m/s · {:.2} km",
            crate::time(p.t_cpa_s),
            p.speed_mps,
            p.distance_m / 1000.0
        ),
        None => m.label().into(),
    };
    match c.rms_cents() {
        Some(r) => format!(
            "{what} · {} points, {r:.0} cents",
            c.points.len() + c.traced.len()
        ),
        None => what,
    }
}

/// One tone of a curve between times a and b, as screen lines (split where it leaves the drawable range).
fn line(
    p: &egui::Painter,
    v: &View,
    c: &Curve,
    tone: f64,
    (a, b): (f64, f64),
    stroke: egui::Stroke,
    dashed: bool,
) {
    let (a, b) = (a.max(v.t0), b.min(v.t1));
    if b <= a {
        return;
    }
    let n = (v.rect.width() as usize).clamp(50, 1500);
    let mut run: Vec<Pos2> = vec![];
    let flush = |run: &mut Vec<Pos2>| {
        if run.len() > 1 {
            if dashed {
                p.extend(egui::Shape::dashed_line(run, stroke, 6.0, 4.0));
            } else {
                p.add(egui::Shape::line(std::mem::take(run), stroke));
            }
        }
        run.clear();
    };
    for k in 0..=n {
        let t = a + (b - a) * k as f64 / n as f64;
        match v
            .pos(t, c.eval(t, tone))
            .filter(|q| q.y >= v.rect.top() - 2.0 && q.y <= v.rect.bottom() + 2.0)
        {
            Some(q) => run.push(q),
            None => flush(&mut run),
        }
    }
    flush(&mut run);
}
