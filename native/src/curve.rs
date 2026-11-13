//! Hand-fitted curves: a frequency-vs-time model with named parameters, set by hand and polished by a
//! fit to points the user places on the spectrogram. Pure maths (no UI), so any model can be added:
//! implement `Model` and list it in `MODELS`.
use serde_json::{json, Value};

/// One adjustable number of a model.
pub struct Param {
    pub key: &'static str,
    pub label: &'static str,
    pub unit: &'static str,
    pub min: f64,
    pub max: f64,
    /// Adjusted and fitted on a log scale (frequencies, speeds, distances).
    pub log: bool,
    /// A typical adjustment: relative for log parameters (0.02 = 2 %), absolute otherwise.
    /// Sets the slider speed and the fit's first steps.
    pub step: f64,
    pub default: f64,
    /// A new curve starts with it locked (left out of the fit until the user frees it).
    pub locked: bool,
}

/// A point the user put on a line of the spectrogram; `tone` = which multiple of the curve it lies on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub t: f64,
    pub f: f64,
    pub tone: f64,
}

/// What a curve says about a straight, steady pass (the models that describe one).
pub struct Pass {
    pub t_cpa_s: f64,
    pub f0_hz: f64,
    pub speed_mps: f64,
    pub distance_m: f64,
}

/// The view the user is looking at, for starting a curve where they are looking.
#[derive(Clone, Copy)]
pub struct Window {
    pub t0: f64,
    pub t1: f64,
    pub fmin: f64,
    pub fmax: f64,
}

pub trait Model: Sync {
    fn key(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn params(&self) -> &'static [Param];
    /// Frequency (Hz) heard at time t (s) of the main tone.
    fn eval(&self, p: &[f64], t: f64) -> f64;
    /// Parameters to start from: from the points when there are some, else from the view.
    fn guess(&self, points: &[Point], view: Window) -> Vec<f64>;
    fn pass(&self, _p: &[f64]) -> Option<Pass> {
        None
    }
}

// ---- the models -----------------------------------------------------------------------------------

/// A straight, constant-speed pass (doppler/physics.py straight_pass): t_cpa is when the sound emitted at
/// closest approach arrives, where the heard frequency equals the source's.
pub struct DopplerPass;
const DOPPLER: [Param; 5] = [
    Param {
        key: "t_cpa",
        label: "Closest approach (heard)",
        unit: "s",
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
        log: false,
        step: 1.0,
        default: 0.0,
        locked: false,
    },
    Param {
        key: "f0",
        label: "Tone at closest approach",
        unit: "Hz",
        min: 5.0,
        max: 8000.0,
        log: true,
        step: 0.01,
        default: 100.0,
        locked: false,
    },
    Param {
        key: "speed",
        label: "Speed",
        unit: "m/s",
        min: 3.0,
        max: 290.0,
        log: true,
        step: 0.05,
        default: 60.0,
        locked: false,
    },
    Param {
        key: "distance",
        label: "Closest distance",
        unit: "m",
        min: 10.0,
        max: 40000.0,
        log: true,
        step: 0.1,
        default: 1000.0,
        locked: false,
    },
    Param {
        key: "c",
        label: "Speed of sound",
        unit: "m/s",
        min: 300.0,
        max: 360.0,
        log: false,
        step: 1.0,
        default: 340.3,
        locked: true,
    },
];
pub fn straight_pass(t: f64, t_cpa: f64, v: f64, d: f64, c: f64) -> f64 {
    let v = v.min(0.95 * c);
    let (c2, v2) = (c * c, v * v);
    let s_t = t - (t_cpa - d / c);
    let s = (c2 * s_t - (c2 * v2 * s_t * s_t + d * d * (c2 - v2)).sqrt()) / (c2 - v2);
    let r = (d * d + v2 * s * s).sqrt();
    c / (c + v2 * s / r)
}
impl Model for DopplerPass {
    fn key(&self) -> &'static str {
        "doppler_pass"
    }
    fn label(&self) -> &'static str {
        "Aircraft pass (straight, steady speed)"
    }
    fn params(&self) -> &'static [Param] {
        &DOPPLER
    }
    fn eval(&self, p: &[f64], t: f64) -> f64 {
        p[1] * straight_pass(t, p[0], p[2], p[3], p[4])
    }
    fn guess(&self, points: &[Point], view: Window) -> Vec<f64> {
        let c = DOPPLER[4].default;
        let main: Vec<Point> = lowest_tone(points);
        if main.len() < 3 {
            return vec![
                (view.t0 + view.t1) / 2.0,
                (view.fmin * view.fmax).sqrt(),
                60.0,
                1000.0,
                c,
            ];
        }
        // speed from the high and low ends ((c+v)/(c-v)), closest approach where the pitch falls fastest,
        // distance from how long the fall takes (about 2 d / v between its quarter points)
        let n = main.len();
        let k = (n / 6).max(1);
        let hi = median(main[..k].iter().map(|q| q.f / q.tone));
        let lo = median(main[n - k..].iter().map(|q| q.f / q.tone));
        let ratio = (hi / lo).max(1.001);
        let v = (c * (ratio - 1.0) / (ratio + 1.0)).clamp(10.0, 250.0);
        let mid = (hi * lo).sqrt();
        let t_cpa = main
            .iter()
            .min_by(|a, b| {
                ((a.f / a.tone).ln() - mid.ln())
                    .abs()
                    .total_cmp(&((b.f / b.tone).ln() - mid.ln()).abs())
            })
            .map_or(main[n / 2].t, |q| q.t);
        let quarter = |w: f64| {
            main.iter()
                .find(|q| q.f / q.tone <= hi.powf(1.0 - w) * lo.powf(w))
                .map(|q| q.t)
        };
        let d = match (quarter(0.25), quarter(0.75)) {
            (Some(a), Some(b)) if b > a => ((b - a) * v / 2.0).clamp(50.0, 20000.0),
            _ => 1000.0,
        };
        vec![t_cpa, mid, v, d, c]
    }
    fn pass(&self, p: &[f64]) -> Option<Pass> {
        Some(Pass {
            t_cpa_s: p[0],
            f0_hz: p[1],
            speed_mps: p[2],
            distance_m: p[3],
        })
    }
}

/// An aircraft that may turn, speed up or slow down, climb or descend, and whose engine note may drift. With
/// those four at 0 it is exactly the straight pass (same t_cpa, tone, speed and slant distance). Geometry: at
/// the moment of closest approach (emission time 0) the plane flies along x at `speed`, `height` above the
/// microphone and the rest of `distance` to the side; it then turns at a steady rate (positive = away from the
/// listener) with steady acceleration and climb.
/// Each heard time is traced back to its emission time (Newton's method on t = tau + range(tau) / c).
pub struct AircraftPath;
const PATH: [Param; 10] = [
    Param {
        key: "t_cpa",
        label: "Closest approach (heard)",
        unit: "s",
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
        log: false,
        step: 1.0,
        default: 0.0,
        locked: false,
    },
    Param {
        key: "f0",
        label: "Tone at closest approach",
        unit: "Hz",
        min: 5.0,
        max: 8000.0,
        log: true,
        step: 0.01,
        default: 100.0,
        locked: false,
    },
    Param {
        key: "speed",
        label: "Speed at closest approach",
        unit: "m/s",
        min: 3.0,
        max: 290.0,
        log: true,
        step: 0.05,
        default: 60.0,
        locked: false,
    },
    Param {
        key: "distance",
        label: "Closest distance",
        unit: "m",
        min: 10.0,
        max: 40000.0,
        log: true,
        step: 0.1,
        default: 1000.0,
        locked: false,
    },
    Param {
        key: "height",
        label: "Height above you",
        unit: "m",
        min: 0.0,
        max: 12000.0,
        log: false,
        step: 50.0,
        default: 300.0,
        locked: true,
    },
    Param {
        key: "accel",
        label: "Speeding up (−: slowing)",
        unit: "m/s²",
        min: -6.0,
        max: 6.0,
        log: false,
        step: 0.2,
        default: 0.0,
        locked: true,
    },
    Param {
        key: "turn",
        label: "Turn rate (+: away from you)",
        unit: "°/s",
        min: -15.0,
        max: 15.0,
        log: false,
        step: 0.3,
        default: 0.0,
        locked: true,
    },
    Param {
        key: "climb",
        label: "Climb (−: descending)",
        unit: "m/s",
        min: -40.0,
        max: 40.0,
        log: false,
        step: 0.5,
        default: 0.0,
        locked: true,
    },
    Param {
        key: "drift",
        label: "Engine note drift",
        unit: "cents/s",
        min: -40.0,
        max: 40.0,
        log: false,
        step: 0.5,
        default: 0.0,
        locked: true,
    },
    Param {
        key: "c",
        label: "Speed of sound",
        unit: "m/s",
        min: 300.0,
        max: 360.0,
        log: false,
        step: 1.0,
        default: 340.3,
        locked: true,
    },
];
impl AircraftPath {
    /// Position and velocity (x along the flight at closest approach, y away from the listener, z up),
    /// relative to the microphone, u seconds of emission time after closest approach.
    fn state(p: &[f64], u: f64) -> ([f64; 3], [f64; 3]) {
        let (v0, d, a, w, climb) = (p[2], p[3], p[5], p[6].to_radians(), p[7]);
        let h = p[4].min(d);
        let side = (d * d - h * h).sqrt();
        let (x, y) = if w.abs() < 1e-4 {
            (
                v0 * u + a * u * u / 2.0,
                v0 * w * u * u / 2.0 + a * w * u * u * u / 3.0,
            )
        } else {
            let (sn, cs) = (w * u).sin_cos();
            let v = v0 + a * u;
            (
                v * sn / w + a * (cs - 1.0) / (w * w),
                (v0 - v * cs) / w + a * sn / (w * w),
            )
        };
        let v = (v0 + a * u).max(1.0);
        let (sn, cs) = (w * u).sin_cos();
        ([x, side + y, h + climb * u], [v * cs, v * sn, climb])
    }
}
impl Model for AircraftPath {
    fn key(&self) -> &'static str {
        "aircraft_path"
    }
    fn label(&self) -> &'static str {
        "Aircraft path (turns, speed changes, climb)"
    }
    fn params(&self) -> &'static [Param] {
        &PATH
    }
    fn eval(&self, p: &[f64], t: f64) -> f64 {
        let c = p[9];
        let range = |u: f64| {
            let (r, v) = Self::state(p, u);
            let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
            (n, (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / n) // range and its rate
        };
        let target = t - p[0] + range(0.0).0 / c; // heard time minus the emission time of closest approach
                                                  // start from the straight pass's emission time, then Newton: g(u) = u + range(u) / c - target
        let (v, d) = (p[2].min(0.95 * c), p[3]);
        let (c2, v2) = (c * c, v * v);
        let mut u =
            (c2 * target - (c2 * v2 * target * target + d * d * (c2 - v2)).sqrt()) / (c2 - v2);
        for _ in 0..8 {
            let (r, rate) = range(u);
            let step = (u + r / c - target) / (1.0 + rate / c).max(0.05);
            u -= step;
            if step.abs() < 1e-6 {
                break;
            }
        }
        let rate = range(u).1.clamp(-0.95 * c, 0.95 * c);
        p[1] * (p[8] * u / 1200.0).exp2() * c / (c + rate)
    }
    fn guess(&self, points: &[Point], view: Window) -> Vec<f64> {
        let s = DopplerPass.guess(points, view);
        vec![
            s[0],
            s[1],
            s[2],
            s[3],
            (0.5 * s[3]).min(300.0),
            0.0,
            0.0,
            0.0,
            0.0,
            s[4],
        ]
    }
    fn pass(&self, p: &[f64]) -> Option<Pass> {
        // shortcut: locating reads a pass as straight; for a turning plane these are its values at closest approach
        Some(Pass {
            t_cpa_s: p[0],
            f0_hz: p[1],
            speed_mps: p[2],
            distance_m: p[3],
        })
    }
}

/// A steady tone (mains hum, an engine heard from far away).
pub struct Tone;
const TONE: [Param; 1] = [Param {
    key: "f",
    label: "Frequency",
    unit: "Hz",
    min: 5.0,
    max: 8000.0,
    log: true,
    step: 0.01,
    default: 100.0,
    locked: false,
}];
impl Model for Tone {
    fn key(&self) -> &'static str {
        "tone"
    }
    fn label(&self) -> &'static str {
        "Steady tone"
    }
    fn params(&self) -> &'static [Param] {
        &TONE
    }
    fn eval(&self, p: &[f64], _t: f64) -> f64 {
        p[0]
    }
    fn guess(&self, points: &[Point], view: Window) -> Vec<f64> {
        let main = lowest_tone(points);
        vec![if main.is_empty() {
            (view.fmin * view.fmax).sqrt()
        } else {
            median(main.iter().map(|q| q.f / q.tone))
        }]
    }
}

/// A straight line between two (time, frequency) ends, extended beyond them.
pub struct Sweep;
const SWEEP: [Param; 4] = [
    Param {
        key: "t_a",
        label: "Start time",
        unit: "s",
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
        log: false,
        step: 1.0,
        default: 0.0,
        locked: false,
    },
    Param {
        key: "f_a",
        label: "Start frequency",
        unit: "Hz",
        min: 5.0,
        max: 8000.0,
        log: true,
        step: 0.01,
        default: 100.0,
        locked: false,
    },
    Param {
        key: "t_b",
        label: "End time",
        unit: "s",
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
        log: false,
        step: 1.0,
        default: 10.0,
        locked: false,
    },
    Param {
        key: "f_b",
        label: "End frequency",
        unit: "Hz",
        min: 5.0,
        max: 8000.0,
        log: true,
        step: 0.01,
        default: 100.0,
        locked: false,
    },
];
impl Model for Sweep {
    fn key(&self) -> &'static str {
        "sweep"
    }
    fn label(&self) -> &'static str {
        "Straight sweep"
    }
    fn params(&self) -> &'static [Param] {
        &SWEEP
    }
    fn eval(&self, p: &[f64], t: f64) -> f64 {
        let span = if (p[2] - p[0]).abs() < 1e-6 {
            1e-6
        } else {
            p[2] - p[0]
        };
        (p[1] + (p[3] - p[1]) * (t - p[0]) / span).max(1.0)
    }
    fn guess(&self, points: &[Point], view: Window) -> Vec<f64> {
        let main = lowest_tone(points);
        match (main.first(), main.last()) {
            (Some(a), Some(b)) if b.t > a.t => vec![a.t, a.f / a.tone, b.t, b.f / b.tone],
            _ => {
                let f = (view.fmin * view.fmax).sqrt();
                let span = view.t1 - view.t0;
                vec![
                    view.t0 + 0.25 * span,
                    f * 1.1,
                    view.t0 + 0.75 * span,
                    f / 1.1,
                ]
            }
        }
    }
}

/// Every model the tool offers; a new model only needs adding here.
/// The first is where new curves start: the aircraft path, with its extras locked (= a straight pass).
pub static MODELS: &[&dyn Model] = &[&AircraftPath, &DopplerPass, &Tone, &Sweep];

pub fn model_index(key: &str) -> Option<usize> {
    MODELS.iter().position(|m| m.key() == key)
}

// ---- a curve being fitted ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Curve {
    pub model: usize,
    pub params: Vec<f64>,
    /// Parameters the fit leaves alone.
    pub locked: Vec<bool>,
    /// The multiples drawn (and saved) besides the main tone.
    pub harmonics: Vec<f64>,
    pub points: Vec<Point>,
    /// Points found by snapping to the bright line near the curve (replaced at each snap).
    pub traced: Vec<Point>,
    /// Real aircraft carry tones that aren't exact multiples (propeller blade rate vs engine firing):
    /// when set, each tone with points keeps its own pitch offset and only the shape is shared.
    pub free_tones: bool,
    /// Offset (cents) of each tone from its exact multiple, when free_tones.
    pub offsets: Vec<(f64, f64)>,
    /// Time span the curve is saved over (where the line is really heard).
    pub extent: (f64, f64),
}

impl Curve {
    pub fn new(model: usize, points: Vec<Point>, view: Window) -> Self {
        let m = MODELS[model];
        let params = m.guess(&points, view);
        let locked = m.params().iter().map(|p| p.locked).collect();
        let extent = points_span(&points).unwrap_or((
            view.t0 + 0.1 * (view.t1 - view.t0),
            view.t1 - 0.1 * (view.t1 - view.t0),
        ));
        let mut c = Self {
            model,
            params,
            locked,
            harmonics: vec![1.0],
            points,
            traced: vec![],
            free_tones: true,
            offsets: vec![],
            extent,
        };
        c.clamp();
        c.refresh_offsets();
        c
    }
    pub fn model(&self) -> &'static dyn Model {
        MODELS[self.model]
    }
    pub fn clamp(&mut self) {
        let specs = self.model().params();
        for (v, p) in self.params.iter_mut().zip(specs) {
            *v = v.clamp(p.min, p.max);
        }
    }
    /// Change the model, keeping the points.
    /// Change the model, keeping the points and every setting both models share (same key).
    pub fn set_model(&mut self, model: usize, view: Window) {
        let old = self.clone();
        *self = Curve::new(model, old.points.clone(), view);
        (self.harmonics, self.free_tones, self.extent) =
            (old.harmonics.clone(), old.free_tones, old.extent);
        for (i, p) in old.model().params().iter().enumerate() {
            if let Some(j) = self.model().params().iter().position(|q| q.key == p.key) {
                self.params[j] = old.params[i];
                self.locked[j] = old.locked[i];
            }
        }
        self.clamp();
        self.refresh_offsets();
    }
    #[cfg(test)]
    pub fn set(&mut self, key: &str, value: f64) {
        if let Some(i) = self.model().params().iter().position(|p| p.key == key) {
            self.params[i] = value;
            self.clamp();
        }
    }
    /// Frequency of tone `tone` at time t, with its own offset when tones are free.
    pub fn eval(&self, t: f64, tone: f64) -> f64 {
        let cents = self
            .offsets
            .iter()
            .find(|(h, _)| (*h - tone).abs() < 1e-9)
            .map_or(0.0, |(_, c)| *c);
        tone * self.model().eval(&self.params, t) * (cents / 1200.0).exp2()
    }
    /// Each point's distance from its tone, in cents (pitch offsets profiled out when tones are free).
    pub fn residuals(&self) -> Vec<f64> {
        residuals(self.model(), &self.params, &self.data(), self.free_tones).0
    }
    pub fn rms_cents(&self) -> Option<f64> {
        let r = self.residuals();
        (!r.is_empty()).then(|| (r.iter().map(|x| x * x).sum::<f64>() / r.len() as f64).sqrt())
    }
    pub fn refresh_offsets(&mut self) {
        self.offsets = residuals(self.model(), &self.params, &self.data(), self.free_tones).1;
    }
    /// Fit the free settings to the points, in cents. The user's clicks count in full (plain least squares,
    /// weighted up so a long snapped trace can't drown them); snapped points are robust (Huber), since a trace
    /// can catch a neighbouring line. Several starts (where the curve is now, a fresh guess from the points,
    /// and each scale setting x0.6 / x1.6 of both) are tried on up to 240 points, the best is refined on all:
    /// one start can stop in a nearby wrong shape (e.g. too flat at the high end).
    pub fn fit(&mut self) {
        let data = self.data();
        if data.is_empty() {
            return;
        }
        let m = self.model();
        let specs = m.params();
        let free: Vec<usize> = (0..specs.len()).filter(|&i| !self.locked[i]).collect();
        if free.is_empty() {
            return;
        }
        let to_z = |i: usize, v: f64| {
            if specs[i].log {
                v.max(1e-9).ln() / specs[i].step
            } else {
                v / specs[i].step
            }
        };
        let from_z = |i: usize, z: f64| {
            let v = if specs[i].log {
                (z * specs[i].step).exp()
            } else {
                z * specs[i].step
            };
            v.clamp(specs[i].min, specs[i].max)
        };
        let base = self.params.clone();
        let full = |z: &[f64]| {
            let mut p = base.clone();
            for (k, &i) in free.iter().enumerate() {
                p[i] = from_z(i, z[k]);
            }
            p
        };
        let n_user = self.points.len();
        let w_user = (self.traced.len() as f64 / (4.0 * n_user.max(1) as f64)).max(1.0);
        let free_tones = self.free_tones;
        // each point counts for the stretch of line it stands for: a steep fall clicked with a few points weighs as
        // much as the long flat ends clicked with many
        let cost_on = |z: &[f64], pts: &[Point], nu: usize, w: &[f64]| {
            let r = residuals(m, &full(z), pts, free_tones).0;
            r[..nu]
                .iter()
                .zip(w)
                .map(|(x, wi)| 0.5 * w_user * wi * x * x)
                .sum::<f64>()
                + r[nu..]
                    .iter()
                    .zip(&w[nu..])
                    .map(|(x, wi)| wi * huber(&[*x], 30.0))
                    .sum::<f64>()
        };
        // the starts, compared on a thinned copy of the points (the user's first, then the trace)
        let every = (data.len() / 240).max(1);
        let sub: Vec<Point> = data.iter().copied().step_by(every).collect();
        let nu_sub = (0..sub.len()).filter(|k| k * every < n_user).count();
        let w_sub = arc_weights(&sub);
        let cost_sub = |z: &[f64]| cost_on(z, &sub, nu_sub, &w_sub);
        let (t0, t1) = points_span(&data).unwrap_or(self.extent);
        let (f_lo, f_hi) = data.iter().fold((f64::MAX, 0.0_f64), |(a, b), q| {
            (a.min(q.f / q.tone), b.max(q.f / q.tone))
        });
        let mut guess = m.guess(
            &data,
            Window {
                t0,
                t1,
                fmin: f_lo,
                fmax: f_hi,
            },
        );
        for (k, g) in guess.iter_mut().enumerate() {
            if self.locked[k] {
                *g = self.params[k]; // a locked setting stays as the user set it
            }
        }
        let mut starts = vec![];
        for p in [&self.params, &guess] {
            let z: Vec<f64> = free.iter().map(|&i| to_z(i, p[i])).collect();
            for (k, &i) in free.iter().enumerate() {
                if specs[i].log {
                    for f in [0.6_f64, 1.6] {
                        let mut v = z.clone();
                        v[k] += f.ln() / specs[i].step;
                        starts.push(v);
                    }
                }
            }
            starts.push(z);
        }
        let mut best = starts
            .iter()
            .map(|z| {
                let z = nelder_mead(&cost_sub, z, 200 * z.len());
                (cost_sub(&z), z)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, z)| z)
            .unwrap();
        let w_all = arc_weights(&data);
        let cost = |z: &[f64]| cost_on(z, &data, n_user, &w_all);
        for _ in 0..3 {
            best = nelder_mead(&cost, &best, 300 * best.len());
        }
        self.params = full(&best);
        self.refresh_offsets();
    }
    /// Free locked settings one at a time (except the speed of sound), each time the one that brings the curve
    /// closest to the points, while it cuts the misfit by a fifth or more (a steep pass with sloping ends can need
    /// two: e.g. engine drift, then speeding up). Returns (their labels, RMS before, RMS after), or None when none
    /// helps that much.
    pub fn free_what_bends_it(&mut self) -> Option<(String, f64, f64)> {
        let before = self.rms_cents()?;
        let specs = self.model().params();
        let mut freed = vec![];
        let mut now = before;
        loop {
            let best = (0..specs.len())
                .filter(|&i| self.locked[i] && specs[i].key != "c")
                .filter_map(|i| {
                    let mut c = self.clone();
                    c.locked[i] = false;
                    c.fit();
                    Some((c.rms_cents()?, i, c))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            match best {
                Some((rms, i, c)) if rms <= 0.8 * now => {
                    (*self, now) = (c, rms);
                    freed.push(specs[i].label);
                }
                _ => break,
            }
        }
        (!freed.is_empty()).then(|| (freed.join("”, “"), before, now))
    }
    /// Start the curve over from the points (first guess from them, then a fit), keeping the model, the
    /// free / locked choice, the tones and their setting; the span becomes the points'.
    pub fn from_points(&mut self, view: Window) {
        let old = self.clone();
        *self = Curve::new(old.model, old.data(), view);
        (self.locked, self.harmonics, self.free_tones) =
            (old.locked, old.harmonics, old.free_tones);
        self.fit();
    }
    /// The user's points and the traced ones: what the fit follows.
    pub fn data(&self) -> Vec<Point> {
        self.points.iter().chain(&self.traced).copied().collect()
    }
    /// Move the curve dt seconds and by a frequency ratio, for any model: every time setting (unit s) moves,
    /// every frequency setting (unit Hz) scales.
    pub fn shift(&mut self, dt: f64, ratio: f64) {
        let specs = self.model().params();
        for (v, p) in self.params.iter_mut().zip(specs) {
            match p.unit {
                "s" => *v += dt,
                "Hz" => *v *= ratio,
                _ => {}
            }
        }
        self.extent = (self.extent.0 + dt, self.extent.1 + dt);
        self.clamp();
    }
    /// Snap to the line: along each tone shown, over the extent, look for the brightest spot within a band
    /// around the curve (brightness 0-1 from `bright(t, f)`, None where nothing is loaded), keep the spots that
    /// stand out from the band, fit, and repeat in narrower bands. Returns the number of traced points
    /// (the previous trace is kept when too little stands out).
    pub fn snap(&mut self, bright: &dyn Fn(f64, f64) -> Option<f32>) -> usize {
        let tones = self.tones();
        for band in [120.0, 60.0, 30.0] {
            let (a, b) = self.extent;
            let n = (((b - a) / 0.25) as usize).clamp(20, 1200);
            let mut found = vec![];
            for &h in &tones {
                for k in 0..=n {
                    let t = a + (b - a) * k as f64 / n as f64;
                    let fc = self.eval(t, h);
                    let mut vals = vec![];
                    let mut best = (f32::MIN, 0.0);
                    for j in -20..=20 {
                        let c = band * j as f64 / 20.0;
                        if let Some(v) = bright(t, fc * (c / 1200.0).exp2()) {
                            vals.push(v);
                            if v > best.0 {
                                best = (v, c);
                            }
                        }
                    }
                    if vals.len() < 15 || best.1.abs() >= 0.95 * band {
                        continue; // not loaded here, or the brightest spot is at the band's edge (the line is beyond)
                    }
                    vals.sort_by(f32::total_cmp);
                    if best.0 - vals[vals.len() / 2] >= SNAP_CONTRAST {
                        found.push(Point {
                            t,
                            f: fc * (best.1 / 1200.0).exp2(),
                            tone: h,
                        });
                    }
                }
            }
            if found.len() < 8 {
                break;
            }
            self.traced = found;
            self.fit();
        }
        self.traced.len()
    }
    pub fn extent_from_points(&mut self) {
        if let Some(e) = points_span(&self.data()) {
            self.extent = e;
        }
    }
    /// The main tone and every harmonic drawn, as lines over the extent (one point per analysis frame).
    pub fn contours(&self) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        let (a, b) = self.extent;
        let n = (((b - a) / 0.125).round() as usize).clamp(2, 200_000);
        self.tones()
            .into_iter()
            .map(|h| {
                let (mut ts, mut fs) = (vec![], vec![]);
                for k in 0..=n {
                    let t = a + (b - a) * k as f64 / n as f64;
                    let f = self.eval(t, h);
                    if f.is_finite() && (15.0..=7500.0).contains(&f) {
                        ts.push((t * 1000.0).round() / 1000.0);
                        fs.push((f * 100.0).round() / 100.0);
                    }
                }
                (h, ts, fs)
            })
            .filter(|(_, ts, _)| ts.len() >= 2)
            .collect()
    }
    /// The tones shown: the chosen harmonics and every tone that has points.
    pub fn tones(&self) -> Vec<f64> {
        let mut hs = self.harmonics.clone();
        hs.extend(self.data().iter().map(|q| q.tone));
        hs.sort_by(f64::total_cmp);
        hs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        hs
    }
    /// The curve as the service stores it (manual.json): parameters to edit it again, and its lines.
    pub fn to_json(&self) -> Value {
        let m = self.model();
        let params: serde_json::Map<String, Value> = m
            .params()
            .iter()
            .zip(&self.params)
            .map(|(p, v)| (p.key.into(), json!(v)))
            .collect();
        json!({
            "model": m.key(),
            "params": params,
            "locked": self.locked,
            "harmonics": self.harmonics,
            "free_tones": self.free_tones,
            "offsets_cents": self.offsets.iter().map(|(h, c)| json!([h, c])).collect::<Vec<_>>(),
            "points": self.points.iter().map(|q| json!([q.t, q.f, q.tone])).collect::<Vec<_>>(),
            "traced": self.traced.iter().map(|q| json!([q.t, q.f, q.tone])).collect::<Vec<_>>(),
            "extent": [self.extent.0, self.extent.1],
            "residual_cents": self.rms_cents(),
            "contours": self.contours().into_iter().map(|(h, t, f)| json!({"tone": h, "t_s": t, "f_hz": f})).collect::<Vec<_>>(),
            "pass": m.pass(&self.params).map(|p| json!({"t_cpa_s": p.t_cpa_s, "f0_hz": p.f0_hz, "speed_mps": p.speed_mps, "distance_m": p.distance_m})),
        })
    }
    pub fn from_json(v: &Value) -> Option<Self> {
        let model = model_index(v["model"].as_str()?)?;
        let specs = MODELS[model].params();
        let params = specs
            .iter()
            .map(|p| v["params"][p.key].as_f64().unwrap_or(p.default))
            .collect();
        let mut locked: Vec<bool> = v["locked"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).collect())
            .unwrap_or_default();
        locked.resize(specs.len(), false);
        let nums = |x: &Value| {
            x.as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        let read = |x: &Value| -> Vec<Point> {
            x.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|q| {
                            Some(Point {
                                t: q[0].as_f64()?,
                                f: q[1].as_f64()?,
                                tone: q[2].as_f64()?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let points = read(&v["points"]);
        let traced = read(&v["traced"]);
        let e = nums(&v["extent"]);
        let mut c = Self {
            model,
            params,
            locked,
            harmonics: nums(&v["harmonics"]),
            points,
            traced,
            free_tones: v["free_tones"].as_bool().unwrap_or(true),
            offsets: vec![],
            extent: (*e.first()?, *e.get(1)?),
        };
        c.clamp();
        c.refresh_offsets();
        Some(c)
    }
}

/// The brightest continuous line inside a time x frequency box, as (time, frequency) points. A best path through the
/// box: brightness summed over its columns, minus `smooth` x (jump / 10 cents)² x 0.01 per step, at most 1500 cents/s;
/// kept where the line stands out from its column's median by at least `sensitivity` (0-1 brightness).
/// `bright(t, f)`: the loaded spectrogram's brightness, None where nothing is loaded.
pub fn trace_box(
    bright: &dyn Fn(f64, f64) -> Option<f32>,
    (t0, t1): (f64, f64),
    (f0, f1): (f64, f64),
    smooth: f64,
    sensitivity: f32,
) -> Vec<(f64, f64)> {
    let (t0, t1, f0, f1) = (
        t0.min(t1),
        t0.max(t1),
        f0.min(f1).max(1.0),
        f0.max(f1).max(1.0),
    );
    let cents = 1200.0 * (f1 / f0).log2();
    if t1 - t0 < 0.5 || cents < 8.0 {
        return vec![];
    }
    let row = (cents / 400.0).max(2.0);
    let rows = (cents / row).ceil() as usize + 1;
    let step = ((t1 - t0) / 150.0).max(0.25);
    let cols = ((t1 - t0) / step).floor() as usize + 1;
    let freq = |r: usize| f0 * (r as f64 * row / 1200.0).exp2();
    let reach = ((1500.0 * step / row).ceil() as usize).max(1);
    let mut b = vec![0f32; cols * rows];
    let mut loaded = vec![false; cols];
    for k in 0..cols {
        for r in 0..rows {
            if let Some(v) = bright(t0 + k as f64 * step, freq(r)) {
                b[k * rows + r] = v;
                loaded[k] = true;
            }
        }
    }
    let penalty = |d: usize| smooth.max(0.0) * 0.01 * (d as f64 * row / 10.0).powi(2);
    let mut score: Vec<f64> = b[..rows].iter().map(|&v| v as f64).collect();
    let mut back = vec![0usize; cols * rows];
    for k in 1..cols {
        let mut next = vec![f64::NEG_INFINITY; rows];
        for (r, best) in next.iter_mut().enumerate() {
            for q in r.saturating_sub(reach)..=(r + reach).min(rows - 1) {
                let v = score[q] - penalty(q.abs_diff(r));
                if v > *best {
                    *best = v;
                    back[k * rows + r] = q;
                }
            }
            *best += b[k * rows + r] as f64;
        }
        score = next;
    }
    let mut r = (0..rows)
        .max_by(|&a, &c| score[a].total_cmp(&score[c]))
        .unwrap_or(0);
    let mut path = vec![0usize; cols];
    for k in (0..cols).rev() {
        path[k] = r;
        r = back[k * rows + r];
    }
    (0..cols)
        .filter(|&k| {
            loaded[k] && {
                let mut col = b[k * rows..(k + 1) * rows].to_vec();
                col.sort_by(f32::total_cmp);
                b[k * rows + path[k]] - col[rows / 2] >= sensitivity
            }
        })
        .map(|k| (t0 + k as f64 * step, freq(path[k])))
        .collect()
}

// ---- helpers ------------------------------------------------------------------------------------------

/// How much brighter than the band's median a spot must be to count as the line (0-1 brightness; the
/// spectrogram's colours span about 21 dB, so ~0.15 is ~3 dB).
const SNAP_CONTRAST: f32 = 0.15;

fn median(xs: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = xs.collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The points on the lowest tone that has any, in time order (the guesses read one line).
fn lowest_tone(points: &[Point]) -> Vec<Point> {
    let Some(h) = points.iter().map(|q| q.tone).min_by(f64::total_cmp) else {
        return vec![];
    };
    let mut v: Vec<Point> = points
        .iter()
        .copied()
        .filter(|q| (q.tone - h).abs() < 1e-9)
        .collect();
    v.sort_by(|a, b| a.t.total_cmp(&b.t));
    v
}

fn points_span(points: &[Point]) -> Option<(f64, f64)> {
    let a = points.iter().map(|q| q.t).min_by(f64::total_cmp)?;
    let b = points.iter().map(|q| q.t).max_by(f64::total_cmp)?;
    (b > a).then_some((a, b))
}

/// Residuals in cents and, when tones are free, each tone's median offset (the lowest tone anchors the
/// frequency, so its offset stays 0).
fn residuals(
    m: &dyn Model,
    p: &[f64],
    points: &[Point],
    free_tones: bool,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let mut r: Vec<f64> = points
        .iter()
        .map(|q| 1200.0 * (q.f / (q.tone * m.eval(p, q.t))).log2())
        .collect();
    for x in r.iter_mut() {
        if !x.is_finite() {
            *x = 2400.0;
        }
    }
    let mut offsets = vec![];
    if free_tones {
        let anchor = points.iter().map(|q| q.tone).min_by(f64::total_cmp);
        let mut tones: Vec<f64> = points.iter().map(|q| q.tone).collect();
        tones.sort_by(f64::total_cmp);
        tones.dedup();
        for h in tones {
            if Some(h) == anchor {
                continue;
            }
            let idx: Vec<usize> = (0..points.len()).filter(|&i| points[i].tone == h).collect();
            let off = median(idx.iter().map(|&i| r[i]));
            for &i in &idx {
                r[i] -= off;
            }
            offsets.push((h, off));
        }
    }
    (r, offsets)
}

/// Each point's share of its line: half the distance to its neighbours on the same tone (1 s ~ 10 cents), relative to
/// the mean and kept within 0.2-5 (a lone point can't dominate).
fn arc_weights(points: &[Point]) -> Vec<f64> {
    let mut w = vec![1.0; points.len()];
    let mut tones: Vec<f64> = points.iter().map(|q| q.tone).collect();
    tones.sort_by(f64::total_cmp);
    tones.dedup();
    for h in tones {
        let mut idx: Vec<usize> = (0..points.len()).filter(|&i| points[i].tone == h).collect();
        idx.sort_by(|&a, &b| points[a].t.total_cmp(&points[b].t));
        let gap = |a: usize, b: usize| {
            (points[b].t - points[a].t).hypot(120.0 * (points[b].f / points[a].f).log2())
        };
        for k in 0..idx.len() {
            let before = if k > 0 { gap(idx[k - 1], idx[k]) } else { 0.0 };
            let after = if k + 1 < idx.len() {
                gap(idx[k], idx[k + 1])
            } else {
                0.0
            };
            w[idx[k]] = (before + after) / 2.0;
        }
    }
    let mean = w.iter().sum::<f64>() / w.len().max(1) as f64;
    w.iter()
        .map(|x| {
            if mean > 0.0 {
                (x / mean).clamp(0.2, 5.0)
            } else {
                1.0
            }
        })
        .collect()
}

fn huber(r: &[f64], delta: f64) -> f64 {
    r.iter()
        .map(|x| {
            if x.abs() <= delta {
                0.5 * x * x
            } else {
                delta * (x.abs() - 0.5 * delta)
            }
        })
        .sum()
}

/// Nelder-Mead minimisation from x0 with unit first steps (the caller scales its variables).
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], max_evals: usize) -> Vec<f64> {
    let n = x0.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = (0..=n)
        .map(|i| {
            let mut x = x0.to_vec();
            if i > 0 {
                x[i - 1] += 1.0;
            }
            let fx = f(&x);
            (x, fx)
        })
        .collect();
    let mut evals = n + 1;
    while evals < max_evals {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        if (simplex[n].1 - simplex[0].1).abs() <= 1e-10 * (1.0 + simplex[0].1.abs()) {
            break;
        }
        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|s| s.0[j]).sum::<f64>() / n as f64)
            .collect();
        let towards = |k: f64| -> Vec<f64> {
            (0..n)
                .map(|j| centroid[j] + k * (simplex[n].0[j] - centroid[j]))
                .collect()
        };
        let xr = towards(-1.0);
        let fr = f(&xr);
        evals += 1;
        if fr < simplex[0].1 {
            let xe = towards(-2.0);
            let fe = f(&xe);
            evals += 1;
            simplex[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (xr, fr);
        } else {
            let xc = towards(if fr < simplex[n].1 { -0.5 } else { 0.5 });
            let fc = f(&xc);
            evals += 1;
            if fc < fr.min(simplex[n].1) {
                simplex[n] = (xc, fc);
            } else {
                let best = simplex[0].0.clone();
                for s in simplex.iter_mut().skip(1) {
                    s.0 = (0..n).map(|j| best[j] + 0.5 * (s.0[j] - best[j])).collect();
                    s.1 = f(&s.0);
                }
                evals += n;
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex.swap_remove(0).0
}

#[cfg(test)]
mod tests {
    use super::*;
    const VIEW: Window = Window {
        t0: 0.0,
        t1: 120.0,
        fmin: 40.0,
        fmax: 800.0,
    };

    fn pass_points(p: &[f64], tones: &[(f64, f64)]) -> Vec<Point> {
        let mut out = vec![];
        for &(h, cents) in tones {
            for k in 0..240 {
                let t = 30.0 + k as f64 * 0.25;
                let wobble = 4.0 * ((k as f64) * 1.7).sin(); // a few cents of tracing noise
                out.push(Point {
                    t,
                    f: h * DopplerPass.eval(p, t) * ((cents + wobble) / 1200.0).exp2(),
                    tone: h,
                });
            }
        }
        out
    }

    #[test]
    fn pass_hears_the_source_tone_at_closest_approach_and_doppler_limits_far_away() {
        let p = [60.0, 100.0, 70.0, 800.0, 340.0];
        assert!((DopplerPass.eval(&p, 60.0) - 100.0).abs() < 1e-9);
        assert!((DopplerPass.eval(&p, -1e5) - 100.0 * 340.0 / 270.0).abs() < 0.01);
        assert!((DopplerPass.eval(&p, 1e5) - 100.0 * 340.0 / 410.0).abs() < 0.01);
    }

    #[test]
    fn fit_recovers_a_pass_from_two_tones_that_are_not_exact_multiples() {
        let truth = [60.0, 96.0, 65.0, 1200.0, 340.3];
        let mut c = Curve::new(
            model_index("doppler_pass").unwrap(),
            pass_points(&truth, &[(1.0, 0.0), (2.0, 35.0)]),
            VIEW,
        );
        c.fit();
        assert!((c.params[0] - 60.0).abs() < 0.5, "t_cpa {}", c.params[0]);
        assert!(
            (c.params[2] / 65.0 - 1.0).abs() < 0.05,
            "speed {}",
            c.params[2]
        );
        assert!(
            (c.params[3] / 1200.0 - 1.0).abs() < 0.1,
            "distance {}",
            c.params[3]
        );
        assert!(
            (c.offsets[0].1 - 35.0).abs() < 3.0,
            "offset {:?}",
            c.offsets
        );
        assert!(c.rms_cents().unwrap() < 6.0);
    }

    #[test]
    fn aircraft_path_is_the_straight_pass_when_it_neither_turns_nor_speeds_up() {
        let straight = [60.0, 100.0, 70.0, 800.0, 340.0];
        let path = [60.0, 100.0, 70.0, 800.0, 500.0, 0.0, 0.0, 0.0, 0.0, 340.0];
        for t in [0.0, 40.0, 55.0, 60.0, 63.0, 80.0, 200.0] {
            assert!(
                (AircraftPath.eval(&path, t) / DopplerPass.eval(&straight, t) - 1.0).abs() < 1e-6,
                "t {t}"
            );
        }
        // a turn bends the far end of the curve: still the source tone at closest approach
        let turning = [60.0, 100.0, 70.0, 800.0, 500.0, 0.0, 3.0, 0.0, 0.0, 340.0];
        assert!((AircraftPath.eval(&turning, 60.0) - 100.0).abs() < 1e-6);
        assert!(
            (AircraftPath.eval(&turning, 120.0) - DopplerPass.eval(&straight, 120.0)).abs() > 0.5
        );
    }

    #[test]
    fn snap_finds_the_line_from_a_rough_start() {
        let truth = [60.0, 120.0, 70.0, 900.0, 340.3];
        // a bright line along the true curve (and its 2nd harmonic) on a dark background
        let bright = |t: f64, f: f64| -> Option<f32> {
            let f1 = DopplerPass.eval(&truth, t);
            let cents = (1200.0 * (f / f1).log2())
                .abs()
                .min((1200.0 * (f / (2.0 * f1)).log2()).abs());
            Some(0.1 + 0.8 * (-(cents / 12.0).powi(2)).exp() as f32)
        };
        let mut c = Curve::new(model_index("doppler_pass").unwrap(), vec![], VIEW);
        c.params = vec![57.0, 126.0, 60.0, 1100.0, 340.3]; // dragged roughly onto it
        c.extent = (35.0, 85.0);
        c.harmonics = vec![1.0, 2.0];
        assert!(c.snap(&bright) > 100);
        assert!(
            (c.params[0] - 60.0).abs() < 0.5 && (c.params[2] / 70.0 - 1.0).abs() < 0.05,
            "{:?}",
            c.params
        );
    }

    #[test]
    fn a_turning_pass_is_fitted_once_the_turn_is_freed() {
        let path = model_index("aircraft_path").unwrap();
        let truth = [60.0, 110.0, 60.0, 900.0, 300.0, 0.0, 2.5, 0.0, 0.0, 340.3];
        let points: Vec<Point> = (0..60)
            .map(|k| 30.0 + k as f64)
            .map(|t| Point {
                t,
                f: AircraftPath.eval(&truth, t),
                tone: 1.0,
            })
            .collect();
        let mut c = Curve::new(path, points, VIEW);
        c.set("height", 300.0); // locked: as the truth
        c.fit();
        let straight = c.rms_cents().unwrap();
        let (label, _, after) = c
            .free_what_bends_it()
            .expect("freeing the turn should help");
        assert!(
            after < 0.67 * straight && after < 3.0,
            "{label}: {straight} -> {after}"
        );
    }

    #[test]
    fn a_box_around_a_line_fills_it_with_points_on_the_line() {
        let truth = [60.0, 120.0, 70.0, 900.0, 340.3];
        // the line, a fainter one 150 cents below, and speckle
        let bright = |t: f64, f: f64| -> Option<f32> {
            let s = 1200.0 * (f / DopplerPass.eval(&truth, t)).log2();
            let (c, c2) = (s.abs(), (s + 150.0).abs());
            let speck = 0.1 * ((t * 7.3 + f * 0.37).sin() * 0.5 + 0.5);
            Some(
                (0.1 + speck
                    + 0.8 * (-(c / 10.0).powi(2)).exp()
                    + 0.3 * (-(c2 / 10.0).powi(2)).exp()) as f32,
            )
        };
        let pts = trace_box(&bright, (40.0, 80.0), (90.0, 160.0), 1.0, 0.15);
        assert!(pts.len() > 100, "{} points", pts.len());
        let worst = pts
            .iter()
            .map(|&(t, f)| (1200.0 * (f / DopplerPass.eval(&truth, t)).log2()).abs())
            .fold(0.0, f64::max);
        assert!(worst < 12.0, "worst point {worst} cents off the line");
    }

    #[test]
    fn dragging_moves_times_and_scales_frequencies_for_any_model() {
        let mut c = Curve::new(model_index("sweep").unwrap(), vec![], VIEW);
        c.params = vec![10.0, 200.0, 20.0, 100.0];
        c.shift(5.0, 1.5);
        assert_eq!(c.params, vec![15.0, 300.0, 25.0, 150.0]);
    }

    #[test]
    fn a_steep_fall_clicked_sparsely_counts_as_much_as_its_flat_ends() {
        // 20 points on each flat end over 40 s, 3 points on a 600-cent fall in 4 s
        let mut pts: Vec<Point> = (0..20)
            .map(|k| Point {
                t: k as f64 * 2.0,
                f: 200.0,
                tone: 1.0,
            })
            .collect();
        pts.extend((1..=3).map(|k| Point {
            t: 40.0 + k as f64,
            f: 200.0 * (-(k as f64) / 8.0).exp2(),
            tone: 1.0,
        }));
        pts.extend((0..20).map(|k| Point {
            t: 44.0 + k as f64 * 2.0,
            f: 200.0 * 0.5_f64.sqrt(),
            tone: 1.0,
        }));
        let w = arc_weights(&pts);
        assert!(w[21] > 3.0 * w[5], "steep {} flat {}", w[21], w[5]);
    }

    #[test]
    fn saved_curve_reads_back_the_same() {
        let mut c = Curve::new(
            model_index("sweep").unwrap(),
            vec![
                Point {
                    t: 5.0,
                    f: 200.0,
                    tone: 1.0,
                },
                Point {
                    t: 9.0,
                    f: 180.0,
                    tone: 1.0,
                },
            ],
            VIEW,
        );
        c.harmonics = vec![1.0, 2.0];
        let back = Curve::from_json(&c.to_json()).unwrap();
        assert_eq!(
            (
                back.model,
                &back.params,
                &back.points,
                &back.harmonics,
                back.extent
            ),
            (c.model, &c.params, &c.points, &c.harmonics, c.extent)
        );
        assert_eq!(c.to_json()["contours"].as_array().unwrap().len(), 2);
    }
}
