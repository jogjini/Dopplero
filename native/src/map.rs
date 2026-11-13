use crate::api::{self, Event};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, Vec2};
use std::{
    collections::{HashMap, HashSet},
    sync::mpsc::Sender,
};

pub struct Map {
    center: (f64, f64),
    /// Continuous zoom level (tiles are fetched at the nearest whole level and scaled).
    zoom: f64,
    pub bounds: Option<[f64; 4]>,
    pub drawing: bool,
    start: Option<Pos2>,
    tiles: HashMap<String, TextureHandle>,
    pending: HashSet<String>,
    failed: HashSet<String>,
}
impl Default for Map {
    fn default() -> Self {
        Self {
            center: (0.5, 0.5),
            zoom: 2.0,
            bounds: None,
            drawing: false,
            start: None,
            tiles: HashMap::new(),
            pending: HashSet::new(),
            failed: HashSet::new(),
        }
    }
}
fn world(lat: f64, lon: f64) -> (f64, f64) {
    let lat = lat.clamp(-85.0, 85.0).to_radians();
    (
        (lon + 180.0) / 360.0,
        (1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0,
    )
}
fn geo(x: f64, y: f64) -> (f64, f64) {
    (
        (std::f64::consts::PI * (1.0 - 2.0 * y))
            .sinh()
            .atan()
            .to_degrees(),
        x * 360.0 - 180.0,
    )
}
impl Map {
    pub fn tile(&mut self, ctx: &egui::Context, key: String, bytes: Result<Vec<u8>, String>) {
        self.pending.remove(&key);
        if bytes.is_err() {
            self.failed.insert(key.clone());
        }
        if let Ok(bytes) = bytes {
            if let Ok(im) = image::load_from_memory(&bytes) {
                let im = im.to_rgba8();
                let size = [im.width() as usize, im.height() as usize];
                if self.tiles.len() > 200 {
                    self.tiles.clear();
                }
                self.tiles.insert(
                    key.clone(),
                    ctx.load_texture(
                        key,
                        egui::ColorImage::from_rgba_unmultiplied(size, im.as_raw()),
                        Default::default(),
                    ),
                );
            }
        }
    }
    pub fn focus(&mut self, lat: f64, lon: f64) {
        self.center = world(lat, lon);
        self.zoom = 10.0;
    }
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        tx: &Sender<Event>,
        candidates: &serde_json::Value,
        track: &serde_json::Value,
    ) -> Option<usize> {
        let height = ui.available_height().max(260.0);
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height),
            Sense::click_and_drag(),
        );
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 8.0, Color32::from_rgb(220, 228, 227));
        let scale = 256.0 * 2f64.powf(self.zoom);
        let center = self.center;
        let screen = |x: f64, y: f64| {
            Pos2::new(
                rect.center().x + ((x - center.0) * scale) as f32,
                rect.center().y + ((y - center.1) * scale) as f32,
            )
        };
        let at = |pt: Pos2| {
            (
                center.0 + (pt.x - rect.center().x) as f64 / scale,
                center.1 + (pt.y - rect.center().y) as f64 / scale,
            )
        };
        let (x0, y0) = at(rect.left_top());
        let (x1, y1) = at(rect.right_bottom());
        let z = self.zoom.round().clamp(1.0, 18.0) as i32;
        let n = 1i32 << z;
        for y in (y0 * n as f64).floor() as i32..=(y1 * n as f64).floor() as i32 {
            if y < 0 || y >= n {
                continue;
            }
            for x in (x0 * n as f64).floor() as i32..=(x1 * n as f64).floor() as i32 {
                if x < 0 || x >= n {
                    continue;
                }
                let key = format!("tile/{z}/{x}/{y}");
                let tl = screen(x as f64 / n as f64, y as f64 / n as f64);
                let br = screen((x + 1) as f64 / n as f64, (y + 1) as f64 / n as f64);
                if let Some(t) = self.tiles.get(&key) {
                    p.image(
                        t.id(),
                        Rect::from_min_max(tl, br),
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                } else {
                    // meanwhile the coarser level's tile, stretched (no blank squares while zooming in)
                    if let Some(t) = self
                        .tiles
                        .get(&format!("tile/{}/{}/{}", z - 1, x / 2, y / 2))
                    {
                        let (u, v) = ((x % 2) as f32 * 0.5, (y % 2) as f32 * 0.5);
                        p.image(
                            t.id(),
                            Rect::from_min_max(tl, br),
                            Rect::from_min_max(Pos2::new(u, v), Pos2::new(u + 0.5, v + 0.5)),
                            Color32::WHITE,
                        );
                    }
                    if !self.failed.contains(&key) && self.pending.insert(key.clone()) {
                        api::image(
                            tx.clone(),
                            key,
                            format!("https://tile.openstreetmap.org/{z}/{x}/{y}.png"),
                            true,
                        );
                    }
                }
            }
        }
        if let Some(b) = self.bounds {
            let a = world(b[1], b[2]);
            let z = world(b[0], b[3]);
            let r = Rect::from_two_pos(screen(a.0, a.1), screen(z.0, z.1));
            p.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(46, 116, 173, 35));
            p.rect_stroke(
                r,
                0.0,
                Stroke::new(2.0_f32, Color32::from_rgb(37, 111, 165)),
                egui::StrokeKind::Inside,
            );
        }
        if let (Some(lats), Some(lons)) = (track["lat"].as_array(), track["lon"].as_array()) {
            let points: Vec<_> = lats
                .iter()
                .zip(lons)
                .filter_map(|(a, b)| Some(world(a.as_f64()?, b.as_f64()?)))
                .map(|(x, y)| screen(x, y))
                .collect();
            if points.len() > 1 {
                p.add(egui::Shape::line(
                    points,
                    Stroke::new(3.0_f32, Color32::from_rgb(25, 106, 180)),
                ));
            }
        }
        let mut selected = None;
        if let Some(top) = candidates.as_array() {
            for (i, c) in top.iter().enumerate().take(10) {
                if let (Some(lat), Some(lon)) = (c["lat"].as_f64(), c["lon"].as_f64()) {
                    let (x, y) = world(lat, lon);
                    let pos = screen(x, y);
                    if rect.contains(pos) {
                        p.circle_filled(
                            pos,
                            11.0,
                            if i == 0 {
                                Color32::from_rgb(198, 66, 55)
                            } else {
                                Color32::from_rgb(208, 131, 47)
                            },
                        );
                        p.text(
                            pos,
                            egui::Align2::CENTER_CENTER,
                            format!("{}", i + 1),
                            egui::FontId::proportional(11.0),
                            Color32::WHITE,
                        );
                        if response.clicked()
                            && response
                                .interact_pointer_pos()
                                .is_some_and(|pt| pt.distance(pos) < 14.0)
                        {
                            selected = Some(i);
                        }
                    }
                }
            }
        }
        if self.drawing {
            if response.drag_started() {
                self.start = response.interact_pointer_pos();
            }
            if let (Some(a), Some(b)) = (self.start, response.interact_pointer_pos()) {
                p.rect_stroke(
                    Rect::from_two_pos(a, b),
                    0.0,
                    Stroke::new(2.0_f32, Color32::from_rgb(37, 111, 165)),
                    egui::StrokeKind::Inside,
                );
            }
            if response.drag_stopped() {
                if let (Some(a), Some(b)) = (self.start.take(), response.interact_pointer_pos()) {
                    if a.distance(b) > 5.0 {
                        let (x0, y0) = at(a);
                        let (x1, y1) = at(b);
                        let (la, lo) = geo(x0.clamp(0.0, 1.0), y0.clamp(0.0, 1.0));
                        let (lb, lob) = geo(x1.clamp(0.0, 1.0), y1.clamp(0.0, 1.0));
                        self.bounds = Some([la.min(lb), la.max(lb), lo.min(lob), lo.max(lob)]);
                        self.drawing = false;
                    }
                }
            }
        } else if response.dragged() {
            let d = ui.input(|i| i.pointer.delta());
            self.center.0 = (self.center.0 - d.x as f64 / scale).clamp(0.0, 1.0);
            self.center.1 = (self.center.1 - d.y as f64 / scale).clamp(0.0, 1.0);
        }
        if response.hovered() {
            // the wheel zooms gradually with how far it turns (a trackpad sends many small steps), a pinch zooms,
            // a horizontal swipe pans; the place under the pointer stays put
            let (scroll, pinch, pointer) = ui.input_mut(|i| {
                let s = i.smooth_scroll_delta;
                i.smooth_scroll_delta = Vec2::ZERO;
                (s, i.zoom_delta(), i.pointer.hover_pos())
            });
            self.center.0 = (self.center.0 - scroll.x as f64 / scale).clamp(0.0, 1.0);
            let dz = scroll.y as f64 * 0.004 + (pinch as f64).log2();
            if dz.abs() > 1e-4 {
                let pos = pointer.unwrap_or(rect.center());
                let (wx, wy) = at(pos);
                self.zoom = (self.zoom + dz).clamp(1.0, 18.0);
                let scale = 256.0 * 2f64.powf(self.zoom);
                self.center.0 = (wx - (pos.x - rect.center().x) as f64 / scale).clamp(0.0, 1.0);
                self.center.1 = (wy - (pos.y - rect.center().y) as f64 / scale).clamp(0.0, 1.0);
            }
        }
        p.text(
            rect.right_bottom() - Vec2::new(7.0, 5.0),
            egui::Align2::RIGHT_BOTTOM,
            "© OpenStreetMap contributors",
            egui::FontId::proportional(11.0),
            Color32::BLACK,
        );
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::{geo, world};
    #[test]
    fn projection_round_trips() {
        for (lat, lon) in [(0.0, 0.0), (60.0, -120.0), (-45.0, 130.0)] {
            let (x, y) = world(lat, lon);
            let (actual_lat, actual_lon) = geo(x, y);
            assert!((actual_lat - lat).abs() < 1e-9);
            assert!((actual_lon - lon).abs() < 1e-9);
        }
    }
}
