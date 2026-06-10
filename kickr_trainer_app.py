"""
KICKR GPX Trainer App  – Enhanced Edition
------------------------------------------
Features:
  • GPX route loading + live Folium map
  • Wahoo KICKR Core BLE/FTMS control (gradient sim + ERG mode)
  • ZWO structured workout support
  • Heart Rate Monitor (Garmin Vivoactive 5 / any BLE HR device)
  • Full workout recording → FIT file export
  • Route screenshot + Workout summary image generation
  • Workout History panel

Dependencies:
    pip install PyQt6 PyQtWebEngine bleak pycycling gpxpy folium numpy \
                fitparse fit-tool Pillow matplotlib
"""

import sys
import os
import asyncio
import threading
import math
import json
import tempfile
import struct
import time
import io
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict

import gpxpy
import folium
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy,
    QProgressBar, QDialog, QFormLayout, QDoubleSpinBox, QSpinBox,
    QComboBox, QDialogButtonBox, QTabWidget, QScrollArea, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QListWidget, QListWidgetItem, QSplitter, QTextEdit, QLineEdit
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, pyqtSlot, QUrl, QThread
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush,
    QLinearGradient, QPixmap, QPolygonF
)
from PyQt6.QtCore import QPointF


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

PROFILE_PATH   = os.path.expanduser("~/.kickr_trainer_profile.json")
WORKOUTS_DIR   = os.path.expanduser("~/KICKRWorkouts")
GARMIN_HR_MAC  = "A0:28:84:0B:8E:4C"
KICKR_MAC      = "D2:9D:40:FB:E7:22"

# BLE Heart Rate Service / Characteristic UUIDs
HR_SERVICE_UUID        = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_CHAR    = "00002a37-0000-1000-8000-00805f9b34fb"

ZONE_COLORS = ["#4a90d9", "#48c774", "#f5a623", "#e05c2a",
               "#e03535", "#9b59b6", "#ff1493"]

# Heart-rate zones (% of max HR, Karvonen-style – simplified)
HR_ZONE_NAMES  = ["Z1 Recovery", "Z2 Aerobic", "Z3 Tempo",
                  "Z4 Threshold", "Z5 VO₂max"]
HR_ZONE_BOUNDS = [(0, 0.60), (0.60, 0.70), (0.70, 0.80),
                  (0.80, 0.90), (0.90, 1.10)]
HR_ZONE_COLORS = ["#4a90d9", "#48c774", "#f5a623", "#e05c2a", "#e03535"]


# ─────────────────────────────────────────────────────────────
#  Cyclist Profile
# ─────────────────────────────────────────────────────────────

@dataclass
class CyclistProfile:
    height_cm: float = 175.0
    weight_kg: float = 75.0
    ftp_watts: int   = 200
    age:       int   = 35
    gender:    str   = "Male"

    ZONE_NAMES  = ["Z1 Active Recovery", "Z2 Endurance", "Z3 Tempo",
                   "Z4 Threshold", "Z5 VO₂max", "Z6 Anaerobic", "Z7 Neuromuscular"]
    ZONE_BOUNDS = [(0, 0.55), (0.55, 0.75), (0.75, 0.90),
                   (0.90, 1.05), (1.05, 1.20), (1.20, 1.50), (1.50, 9.99)]

    def zone_for_pct(self, frac: float) -> Tuple[int, str]:
        for i, (lo, hi) in enumerate(self.ZONE_BOUNDS):
            if lo <= frac < hi:
                return i + 1, self.ZONE_NAMES[i]
        return 7, self.ZONE_NAMES[6]

    def target_watts(self, frac: float) -> int:
        return max(0, int(frac * self.ftp_watts))

    def max_hr(self) -> int:
        return max(100, 220 - self.age)

    def hr_zone(self, bpm: int) -> Tuple[int, str]:
        mhr = self.max_hr()
        frac = bpm / mhr
        for i, (lo, hi) in enumerate(HR_ZONE_BOUNDS):
            if lo <= frac < hi:
                return i + 1, HR_ZONE_NAMES[i]
        return 5, HR_ZONE_NAMES[4]

    def calories_per_second(self, power_w: float, hr_bpm: float = 0.0) -> float:
        """Hybrid calorie estimate: uses HR when available, else power-based."""
        if hr_bpm > 40:
            # Keytel formula (gender-specific)
            if self.gender == "Female":
                kcal_min = ((-20.4022 + 0.4472 * hr_bpm
                             - 0.1263 * self.weight_kg
                             + 0.074  * self.age) / 4.184)
            else:
                kcal_min = ((-55.0969 + 0.6309 * hr_bpm
                             + 0.1988 * self.weight_kg
                             + 0.2017 * self.age) / 4.184)
            return max(0.0, kcal_min / 60.0)
        eff = 0.24 if self.gender != "Female" else 0.22
        return power_w / (eff * 4184)

    def save(self):
        with open(PROFILE_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "CyclistProfile":
        if os.path.exists(PROFILE_PATH):
            try:
                with open(PROFILE_PATH) as f:
                    data = json.load(f)
                obj = cls()
                for k, v in data.items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
                return obj
            except Exception:
                pass
        return cls()


# ─────────────────────────────────────────────────────────────
#  ZWO Parser
# ─────────────────────────────────────────────────────────────

@dataclass
class ZwoSegment:
    kind:           str
    duration_s:     int
    power_frac:     float
    power_frac_end: float = field(default=-1.0)

    @property
    def is_ramp(self) -> bool:
        return self.power_frac_end >= 0


def parse_zwo(path: str) -> Tuple[str, str, List[ZwoSegment]]:
    tree = ET.parse(path)
    root = tree.getroot()
    name = root.findtext("name") or "Unnamed Workout"
    desc = root.findtext("description") or ""
    segments: List[ZwoSegment] = []
    workout_node = root.find("workout")
    if workout_node is None:
        raise ValueError("No <workout> block found in .zwo file.")
    for elem in workout_node:
        tag = elem.tag
        if tag == "SteadyState":
            dur = int(float(elem.attrib.get("Duration", 0)))
            pwr = float(elem.attrib.get("Power", 0))
            segments.append(ZwoSegment("SteadyState", dur, pwr))
        elif tag in ("Warmup", "Cooldown"):
            dur = int(float(elem.attrib.get("Duration", 0)))
            plo = float(elem.attrib.get("PowerLow",  0.3))
            phi = float(elem.attrib.get("PowerHigh", 0.7))
            segments.append(ZwoSegment(tag, dur, plo, phi))
        elif tag == "IntervalsT":
            repeat  = int(elem.attrib.get("Repeat", 1))
            on_dur  = int(float(elem.attrib.get("OnDuration",  60)))
            off_dur = int(float(elem.attrib.get("OffDuration", 90)))
            on_pwr  = float(elem.attrib.get("OnPower",  1.0))
            off_pwr = float(elem.attrib.get("OffPower", 0.5))
            for _ in range(repeat):
                segments.append(ZwoSegment("Interval_On",  on_dur,  on_pwr))
                segments.append(ZwoSegment("Interval_Off", off_dur, off_pwr))
        elif tag == "FreeRide":
            dur = int(float(elem.attrib.get("Duration", 0)))
            segments.append(ZwoSegment("FreeRide", dur, 0.5))
    return name, desc, segments


def total_zwo_duration(segments: List[ZwoSegment]) -> int:
    return sum(s.duration_s for s in segments)


def segment_and_power_at(segments: List[ZwoSegment], elapsed_s: float
                          ) -> Tuple[Optional[ZwoSegment], float]:
    t = 0.0
    for seg in segments:
        if elapsed_s < t + seg.duration_s:
            local = elapsed_s - t
            if seg.is_ramp:
                frac = local / seg.duration_s
                pwr = seg.power_frac + frac * (seg.power_frac_end - seg.power_frac)
            else:
                pwr = seg.power_frac
            return seg, pwr
        t += seg.duration_s
    return None, 0.0


# ─────────────────────────────────────────────────────────────
#  Workout Record  (one data point per second)
# ─────────────────────────────────────────────────────────────

@dataclass
class WorkoutRecord:
    timestamp:  datetime
    lat:        float = 0.0
    lon:        float = 0.0
    altitude:   float = 0.0
    distance:   float = 0.0   # metres cumulative
    speed:      float = 0.0   # km/h
    heart_rate: int   = 0
    cadence:    int   = 0
    power:      int   = 0
    grade:      float = 0.0
    temperature: float = 0.0


@dataclass
class WorkoutSummary:
    start_time:     datetime = field(default_factory=datetime.now)
    end_time:       Optional[datetime] = None
    gpx_name:       str = ""
    zwo_name:       str = ""
    ftp_watts:      int = 0
    weight_kg:      float = 0.0
    total_distance: float = 0.0   # metres
    total_ascent:   float = 0.0
    total_descent:  float = 0.0
    calories:       float = 0.0
    avg_speed:      float = 0.0
    max_speed:      float = 0.0
    avg_power:      float = 0.0
    max_power:      int   = 0
    avg_cadence:    float = 0.0
    max_cadence:    int   = 0
    avg_hr:         float = 0.0
    max_hr:         int   = 0
    np_watts:       float = 0.0   # Normalized Power
    if_value:       float = 0.0   # Intensity Factor
    tss:            float = 0.0   # Training Stress Score
    energy_kj:      float = 0.0
    elapsed_s:      int   = 0
    moving_s:       int   = 0
    records:        List[WorkoutRecord] = field(default_factory=list)

    def compute_from_records(self, ftp: int):
        """Derive all summary stats from the record list."""
        if not self.records:
            return
        recs = self.records
        self.elapsed_s = int((recs[-1].timestamp - recs[0].timestamp).total_seconds())

        speeds   = [r.speed for r in recs if r.speed > 0.5]
        powers   = [r.power for r in recs if r.power > 0]
        cadences = [r.cadence for r in recs if r.cadence > 0]
        hrs      = [r.heart_rate for r in recs if r.heart_rate > 30]

        self.moving_s      = len(speeds)
        self.avg_speed     = sum(speeds) / len(speeds) if speeds else 0
        self.max_speed     = max(speeds) if speeds else 0
        self.avg_power     = sum(powers) / len(powers) if powers else 0
        self.max_power     = max(powers) if powers else 0
        self.avg_cadence   = sum(cadences) / len(cadences) if cadences else 0
        self.max_cadence   = max(cadences) if cadences else 0
        self.avg_hr        = sum(hrs) / len(hrs) if hrs else 0
        self.max_hr        = max(hrs) if hrs else 0
        self.total_distance = recs[-1].distance

        # Elevation stats
        eles = [r.altitude for r in recs if r.altitude > 0]
        asc = desc = 0.0
        for i in range(1, len(eles)):
            d = eles[i] - eles[i-1]
            if d > 0:
                asc += d
            else:
                desc += abs(d)
        self.total_ascent  = asc
        self.total_descent = desc

        # Energy (kJ)
        self.energy_kj = sum(r.power for r in recs) / 1000.0

        # Normalized Power (30-s rolling average, 4th power mean)
        if len(powers) >= 30:
            pw = np.array([r.power for r in recs], dtype=float)
            rolling = np.convolve(pw, np.ones(30)/30, mode='valid')
            self.np_watts = float(np.mean(rolling**4) ** 0.25)
        else:
            self.np_watts = self.avg_power

        # Intensity Factor & TSS
        if ftp > 0:
            self.if_value = self.np_watts / ftp
            self.tss = (self.elapsed_s * self.np_watts * self.if_value) / (ftp * 3600) * 100
        self.ftp_watts = ftp


# ─────────────────────────────────────────────────────────────
#  FIT File Writer  (pure-Python, no external FIT library needed)
# ─────────────────────────────────────────────────────────────

FIT_EPOCH = datetime(1989, 12, 31, 0, 0, 0, tzinfo=timezone.utc)

def to_fit_timestamp(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((dt - FIT_EPOCH).total_seconds())

def _fit_crc(data: bytes) -> int:
    crc_table = [
        0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
        0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
    ]
    crc = 0
    for byte in data:
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc ^= tmp ^ crc_table[byte & 0xF]
        tmp = crc_table[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc ^= tmp ^ crc_table[(byte >> 4) & 0xF]
    return crc

def write_fit_file(summary: WorkoutSummary) -> bytes:
    """
    Write a minimal but valid FIT file containing:
      file_id, device_info, session, lap, and record messages.
    Compatible with Garmin Connect / Strava / TrainingPeaks.
    """
    messages = []

    def def_msg(local_num: int, global_num: int, fields: list) -> bytes:
        """Build a Definition Message."""
        num_fields = len(fields)
        hdr = struct.pack("BBHB",
                          0x40 | local_num,   # definition header
                          0,                   # reserved
                          global_num,          # global message number
                          num_fields)
        field_defs = b"".join(
            struct.pack("BBB", fnum, fsize, ftype)
            for fnum, fsize, ftype in fields
        )
        return hdr + field_defs

    def data_msg(local_num: int, fmt: str, values: list) -> bytes:
        hdr = struct.pack("B", local_num)
        return hdr + struct.pack(fmt, *values)

    # ── File Header (will be written last with correct data_size) ──
    # ── Message 0: file_id (global 0) ──────────────────────────────
    messages.append(def_msg(0, 0, [
        (0, 1, 0),   # type   uint8
        (1, 2, 132), # manufacturer uint16
        (2, 2, 132), # product uint16
        (4, 4, 134), # time_created uint32
        (5, 4, 134), # serial_number uint32
    ]))
    messages.append(data_msg(0, "<BHHII",
        [4,       # activity
         255,     # manufacturer: development
         1,       # product
         to_fit_timestamp(summary.start_time),
         12345678]))

    # ── Message 1: device_info (global 23) ─────────────────────────
    messages.append(def_msg(1, 23, [
        (253, 4, 134),  # timestamp
        (0,   4, 134),  # serial_number
        (2,   2, 132),  # cum_op_time
        (4,   2, 132),  # manufacturer
        (5,   2, 132),  # product
        (3,   2, 132),  # software version
    ]))
    messages.append(data_msg(1, "<IIIHHHH",
        [to_fit_timestamp(summary.start_time),
         12345678, 0, 255, 1, 100, 0]))  # padded to match 7 fields

    # ── Message 2: record (global 20) ──────────────────────────────
    messages.append(def_msg(2, 20, [
        (253, 4, 134),  # timestamp uint32
        (0,   4, 133),  # position_lat sint32
        (1,   4, 133),  # position_long sint32
        (2,   2, 132),  # altitude uint16   (m * 5 + 500)
        (3,   2, 132),  # heart_rate uint8 → stored as uint16 padded
        (4,   1, 2),    # cadence uint8
        (5,   4, 134),  # distance uint32  (m * 100)
        (6,   2, 132),  # speed uint16     (m/s * 1000)
        (7,   2, 132),  # power uint16
        (9,   1, 2),    # grade sint8 (× 2)
    ]))

    def sc(val: float) -> int:
        """semicircles"""
        return int(val * (2**31 / 180))

    for rec in summary.records:
        ts   = to_fit_timestamp(rec.timestamp)
        lat  = sc(rec.lat)  if rec.lat  else 0x7FFFFFFF
        lon  = sc(rec.lon)  if rec.lon  else 0x7FFFFFFF
        alt  = max(0, int(rec.altitude * 5 + 500))
        hr   = min(255, max(0, rec.heart_rate))
        cad  = min(254, max(0, rec.cadence))
        dist = int(rec.distance * 100)
        spd  = int((rec.speed / 3.6) * 1000)   # km/h → mm/s
        pwr  = min(65534, max(0, rec.power))
        grd  = max(-127, min(127, int(rec.grade * 2)))
        messages.append(data_msg(2, "<IiihBIHHb",
            [ts, lat, lon, alt, hr, cad, dist, spd, pwr, grd]))

    # ── Message 3: lap (global 19) ─────────────────────────────────
    messages.append(def_msg(3, 19, [
        (253, 4, 134),  # timestamp
        (2,   4, 134),  # start_time
        (7,   4, 134),  # total_elapsed_time (ms)
        (8,   4, 134),  # total_timer_time (ms)
        (9,   4, 134),  # total_distance (cm)
        (11,  2, 132),  # total_calories
        (13,  2, 132),  # avg_speed (mm/s)
        (14,  2, 132),  # max_speed (mm/s)
        (15,  2, 132),  # avg_power
        (16,  2, 132),  # max_power
        (17,  1, 2),    # avg_cadence
        (18,  1, 2),    # max_cadence
        (19,  1, 2),    # avg_heart_rate
        (20,  1, 2),    # max_heart_rate
    ]))
    ts_end = to_fit_timestamp(summary.end_time or datetime.now(timezone.utc))
    messages.append(data_msg(3, "<IIIIIHHHHHHBBBB",
        [ts_end,
         to_fit_timestamp(summary.start_time),
         summary.elapsed_s * 1000,
         summary.moving_s  * 1000,
         int(summary.total_distance * 100),
         int(summary.calories),
         int((summary.avg_speed / 3.6) * 1000),
         int((summary.max_speed / 3.6) * 1000),
         int(summary.avg_power),
         summary.max_power,
         int(summary.avg_cadence),
         summary.max_cadence,
         int(summary.avg_hr),
         summary.max_hr]))

    # ── Message 4: session (global 18) ─────────────────────────────
    messages.append(def_msg(4, 18, [
        (253, 4, 134),  # timestamp
        (2,   4, 134),  # start_time
        (7,   4, 134),  # total_elapsed_time
        (8,   4, 134),  # total_timer_time
        (9,   4, 134),  # total_distance
        (11,  2, 132),  # total_calories
        (14,  2, 132),  # avg_speed
        (15,  2, 132),  # max_speed
        (16,  2, 132),  # avg_power
        (17,  2, 132),  # max_power
        (19,  1, 2),    # avg_cadence
        (20,  1, 2),    # max_cadence
        (21,  1, 2),    # avg_heart_rate
        (22,  1, 2),    # max_heart_rate
        (0,   1, 0),    # event (0=timer)
        (1,   1, 0),    # event_type (1=stop)
        (5,   1, 0),    # sport (2=cycling)
        (6,   1, 0),    # sub_sport (6=indoor_cycling)
        (25,  2, 132),  # normalized_power
        (26,  2, 132),  # training_stress_score (×10)
        (29,  2, 132),  # intensity_factor (×1000)
        (48,  4, 134),  # total_work (J)
    ]))
    messages.append(data_msg(4, "<IIIIIHHHHHBBBBBBBBHHHl",
        [ts_end,
         to_fit_timestamp(summary.start_time),
         summary.elapsed_s * 1000,
         summary.moving_s  * 1000,
         int(summary.total_distance * 100),
         int(summary.calories),
         int((summary.avg_speed / 3.6) * 1000),
         int((summary.max_speed / 3.6) * 1000),
         int(summary.avg_power),
         summary.max_power,
         int(summary.avg_cadence),
         summary.max_cadence,
         int(summary.avg_hr),
         summary.max_hr,
         0, 1,          # event, event_type
         2, 6,          # sport=cycling, sub=indoor
         int(summary.np_watts),
         int(summary.tss * 10),
         int(summary.if_value * 1000),
         int(summary.energy_kj * 1000)]))  # J

    # ── Message 5: activity (global 34) ────────────────────────────
    messages.append(def_msg(5, 34, [
        (253, 4, 134),
        (1,   4, 134),
        (2,   2, 132),
        (3,   1, 0),
        (4,   1, 0),
    ]))
    messages.append(data_msg(5, "<IIIHBB",
        [ts_end,
         to_fit_timestamp(summary.start_time),
         1,    # num_sessions
         0,    # type=manual
         4]))  # event=activity – pad to 6 values
    # Reassemble with proper field count:
    messages[-1] = struct.pack("B", 5) + struct.pack("<IIIHBB",
        ts_end,
        to_fit_timestamp(summary.start_time),
        summary.elapsed_s * 1000,
        1, 0, 4)

    # ── Assemble data body ─────────────────────────────────
    body = b"".join(messages)
    crc_val = _fit_crc(body)
    body += struct.pack("<H", crc_val)

    # ── File header (14 bytes) ─────────────────────────────
    data_size = len(body)
    hdr = struct.pack("<BBHIHH",
                      14,            # header size
                      0x10,          # protocol version 1.0
                      2100,          # profile version 21.00
                      data_size,
                      0x2E464954,    # ".FIT"
                      _fit_crc(struct.pack("<BBHIH",
                                          14, 0x10, 2100, data_size, 0x2E464954)))

    return hdr + body


# ─────────────────────────────────────────────────────────────
#  Route Screenshot + Summary Image (PIL/Matplotlib)
# ─────────────────────────────────────────────────────────────

def generate_route_image(summary: WorkoutSummary, output_path: str):
    """Render a simple route line image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        recs = [r for r in summary.records if r.lat != 0 and r.lon != 0]
        if not recs:
            _generate_placeholder_route(output_path)
            return

        lats = [r.lat for r in recs]
        lons = [r.lon for r in recs]
        dist_km = summary.total_distance / 1000

        # Color the route by elevation
        eles = np.array([r.altitude for r in recs])

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        # Route gradient coloring
        if len(recs) > 1:
            from matplotlib.collections import LineCollection
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(eles.min(), eles.max())
            lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2.5)
            lc.set_array(eles[:-1])
            ax.add_collection(lc)

        # Start / Finish markers
        ax.plot(lons[0],  lats[0],  'o', color="#00FF88", ms=10, zorder=5, label="Start")
        ax.plot(lons[-1], lats[-1], 's', color="#FF4466", ms=10, zorder=5, label="Finish")

        # Completed portion
        done_idx = int(len(recs) * min(1.0, summary.total_distance /
                                        max(summary.total_distance, 1)))
        ax.plot(lons[:done_idx], lats[:done_idx], color="#FF8C00",
                linewidth=3, alpha=0.7, zorder=3)

        ax.set_xlim(min(lons) - 0.005, max(lons) + 0.005)
        ax.set_ylim(min(lats) - 0.003, max(lats) + 0.003)
        ax.tick_params(colors='#888', labelsize=7)
        ax.spines[:].set_color("#333")

        # Info box
        info = (f"Distance: {dist_km:.2f} km   "
                f"Ascent: {summary.total_ascent:.0f} m   "
                f"Descent: {summary.total_descent:.0f} m")
        ax.set_title(info, color="#FF8C00", fontsize=10, pad=10)
        ax.legend(loc="upper left", facecolor="#222", edgecolor="#444",
                  labelcolor="white", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor="#1a1a1a")
        plt.close(fig)
    except Exception as e:
        _generate_placeholder_route(output_path)


def _generate_placeholder_route(output_path: str):
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (800, 480), color=(26, 26, 26))
        d = ImageDraw.Draw(img)
        d.text((350, 220), "No GPS Data", fill=(100, 100, 100))
        img.save(output_path)
    except Exception:
        pass


def generate_summary_image(summary: WorkoutSummary, output_path: str):
    """Generate a styled workout summary card image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(12, 7), facecolor="#141414")
        gs  = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4,
                       top=0.85, bottom=0.08, left=0.06, right=0.97)

        fig.text(0.5, 0.93, "WORKOUT SUMMARY", ha="center",
                 color="#FF8C00", fontsize=16, fontweight="bold",
                 fontfamily="monospace")

        start_str = summary.start_time.strftime("%d %b %Y  %H:%M")
        dur_s = summary.elapsed_s
        dur_str = f"{dur_s//3600:02d}:{(dur_s%3600)//60:02d}:{dur_s%60:02d}"
        fig.text(0.5, 0.89, f"{start_str}   ·   {dur_str}", ha="center",
                 color="#888", fontsize=9, fontfamily="monospace")

        metrics = [
            ("Distance",    f"{summary.total_distance/1000:.2f}", "km",   "#AAFFAA"),
            ("Avg Speed",   f"{summary.avg_speed:.1f}",          "km/h",  "#FFFFFF"),
            ("Avg Power",   f"{summary.avg_power:.0f}",          "W",     "#FF8C00"),
            ("Max Power",   f"{summary.max_power}",              "W",     "#FF8C00"),
            ("Avg Cadence", f"{summary.avg_cadence:.0f}",        "rpm",   "#00CFFF"),
            ("Avg HR",      f"{summary.avg_hr:.0f}",             "bpm",   "#FF4466"),
            ("Max HR",      f"{summary.max_hr}",                 "bpm",   "#FF4466"),
            ("Calories",    f"{summary.calories:.0f}",           "kcal",  "#FF6B6B"),
            ("Ascent",      f"{summary.total_ascent:.0f}",       "m",     "#48c774"),
            ("Descent",     f"{summary.total_descent:.0f}",      "m",     "#f5a623"),
            ("NP",          f"{summary.np_watts:.0f}",           "W",     "#e05c2a"),
            ("IF",          f"{summary.if_value:.2f}",           "",      "#9b59b6"),
            ("TSS",         f"{summary.tss:.0f}",                "",      "#4a90d9"),
            ("Energy",      f"{summary.energy_kj:.0f}",          "kJ",    "#48c774"),
            ("FTP",         f"{summary.ftp_watts}",              "W",     "#FF8C00"),
            ("Weight",      f"{summary.weight_kg:.1f}",          "kg",    "#AAFFAA"),
        ]

        positions = [(r, c) for r in range(4) for c in range(4)]

        for idx, (label, val, unit, color) in enumerate(metrics):
            if idx >= len(positions):
                break
            row, col = positions[idx]
            if row >= 3:
                break
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1e1e1e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_xticks([]); ax.set_yticks([])

            ax.text(0.5, 0.72, val, ha="center", va="center",
                    color=color, fontsize=18, fontweight="bold",
                    fontfamily="monospace", transform=ax.transAxes)
            ax.text(0.5, 0.38, unit, ha="center", va="center",
                    color="#666", fontsize=8, transform=ax.transAxes)
            ax.text(0.5, 0.12, label.upper(), ha="center", va="center",
                    color="#555", fontsize=7, fontweight="bold",
                    transform=ax.transAxes)

        # ── HR zone bar (bottom) ───────────────────────────
        if summary.records:
            ax_hr = fig.add_axes([0.06, 0.02, 0.56, 0.06])
            ax_hr.set_facecolor("#111")
            hrs = [r.heart_rate for r in summary.records if r.heart_rate > 0]
            if hrs:
                zone_counts = [0] * 5
                mhr = 220 - 35   # approximate
                for h in hrs:
                    frac = h / mhr
                    for zi, (lo, hi) in enumerate(HR_ZONE_BOUNDS):
                        if lo <= frac < hi:
                            zone_counts[zi] += 1
                            break
                total = sum(zone_counts) or 1
                left = 0.0
                for zi, cnt in enumerate(zone_counts):
                    w = cnt / total
                    ax_hr.barh(0, w, left=left, color=HR_ZONE_COLORS[zi],
                               height=0.8, alpha=0.85)
                    if w > 0.05:
                        ax_hr.text(left + w/2, 0, f"Z{zi+1}",
                                   ha="center", va="center",
                                   color="white", fontsize=7, fontweight="bold")
                    left += w
            ax_hr.set_xlim(0, 1); ax_hr.set_ylim(-0.5, 0.5)
            ax_hr.set_xticks([]); ax_hr.set_yticks([])
            ax_hr.set_title("HR Zones", color="#888", fontsize=7, pad=2)
            for s in ax_hr.spines.values():
                s.set_edgecolor("#333")

        # ── Power timeline (bottom right) ──────────────────
        if summary.records:
            ax_pw = fig.add_axes([0.65, 0.02, 0.32, 0.12])
            ax_pw.set_facecolor("#111")
            times  = list(range(len(summary.records)))
            powers = [r.power for r in summary.records]
            ax_pw.fill_between(times, powers, alpha=0.4, color="#FF8C00")
            ax_pw.plot(times, powers, color="#FF8C00", linewidth=0.6)
            ax_pw.set_xlim(0, max(times) if times else 1)
            ax_pw.set_xticks([]); ax_pw.set_yticks([])
            ax_pw.set_title("Power Timeline", color="#888", fontsize=7, pad=2)
            for s in ax_pw.spines.values():
                s.set_edgecolor("#333")

        plt.savefig(output_path, dpi=150, facecolor="#141414")
        plt.close(fig)
    except Exception as e:
        print(f"Summary image error: {e}")


# ─────────────────────────────────────────────────────────────
#  Workout History Storage
# ─────────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    date_str:      str
    fit_path:      str
    route_img:     str
    summary_img:   str
    distance_km:   float
    duration_s:    int
    avg_power:     float
    avg_hr:        float
    calories:      float
    gpx_name:      str
    zwo_name:      str


def load_history() -> List[HistoryEntry]:
    index_path = os.path.join(WORKOUTS_DIR, "history.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path) as f:
            data = json.load(f)
        return [HistoryEntry(**e) for e in data]
    except Exception:
        return []


def save_history(entries: List[HistoryEntry]):
    os.makedirs(WORKOUTS_DIR, exist_ok=True)
    index_path = os.path.join(WORKOUTS_DIR, "history.json")
    with open(index_path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)


# ─────────────────────────────────────────────────────────────
#  Heart Rate BLE Worker
# ─────────────────────────────────────────────────────────────

class HRWorker(QObject):
    """Connects to a BLE heart-rate device and emits HR readings."""
    hr_received     = pyqtSignal(int)           # bpm
    status_changed  = pyqtSignal(str)
    connected       = pyqtSignal(str)           # device address
    disconnected    = pyqtSignal()
    scan_result     = pyqtSignal(list)          # list of (name, address) tuples

    def __init__(self):
        super().__init__()
        self._loop        = None
        self._running     = False
        self._target_addr = GARMIN_HR_MAC
        self._auto_connect = True

    def set_target(self, address: str):
        self._target_addr = address

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def scan(self):
        """Start a BLE scan and emit results."""
        t = threading.Thread(target=self._run_scan, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_stream())

    def _run_scan(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._do_scan())

    async def _do_scan(self):
        try:
            from bleak import BleakScanner
            self.status_changed.emit("Scanning for HR devices…")
            devices = await BleakScanner.discover(timeout=8.0)
            results = [(d.name or "Unknown", d.address) for d in devices
                       if d.name]
            self.scan_result.emit(results)
            self.status_changed.emit(f"Found {len(results)} devices")
        except Exception as e:
            self.status_changed.emit(f"Scan error: {e}")
            self.scan_result.emit([])

    async def _connect_and_stream(self):
        from bleak import BleakClient, BleakError

        # Try preferred address first
        addr = self._target_addr
        self.status_changed.emit(f"Connecting HR: {addr}…")
        try:
            async with BleakClient(addr) as client:
                self.connected.emit(addr)
                self.status_changed.emit(f"HR Connected ● {addr}")

                def hr_handler(sender, data: bytearray):
                    flags = data[0]
                    if flags & 0x01:   # 16-bit HR
                        bpm = struct.unpack_from("<H", data, 1)[0]
                    else:              # 8-bit HR
                        bpm = data[1]
                    self.hr_received.emit(bpm)

                await client.start_notify(HR_MEASUREMENT_CHAR, hr_handler)
                while self._running:
                    await asyncio.sleep(1)
                await client.stop_notify(HR_MEASUREMENT_CHAR)
        except Exception as e:
            self.status_changed.emit(f"HR disconnected: {e}")
            self.disconnected.emit()

    def reconnect(self):
        if not self._running:
            self.start()


# ─────────────────────────────────────────────────────────────
#  KICKR BLE Worker
# ─────────────────────────────────────────────────────────────

class KICKRWorker(QObject):
    data_received   = pyqtSignal(float, float, float, float)
    status_changed  = pyqtSignal(str)
    connected       = pyqtSignal()
    disconnected    = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._loop         = None
        self._ftms         = None
        self._running      = False
        self._target_grade = 0.0
        self._target_power = None
        self._mode         = "sim"

    def set_grade(self, grade_pct: float):
        self._mode         = "sim"
        self._target_grade = max(-20.0, min(20.0, grade_pct))
        if self._loop and self._ftms:
            asyncio.run_coroutine_threadsafe(self._apply_grade(), self._loop)

    def set_power_target(self, watts: int):
        self._mode         = "erg"
        self._target_power = max(10, min(2000, watts))
        if self._loop and self._ftms:
            asyncio.run_coroutine_threadsafe(self._apply_erg(), self._loop)

    def start(self):
        self._running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run())

    async def _connect_and_run(self):
        from bleak import BleakClient
        from pycycling.fitness_machine_service import FitnessMachineService
        self.status_changed.emit("Connecting to KICKR…")
        try:
            async with BleakClient(KICKR_MAC) as client:
                self._ftms = FitnessMachineService(client)
                self._ftms.set_indoor_bike_data_handler(self._handle_data)
                await self._ftms.request_control()
                await self._ftms.enable_indoor_bike_data_notify()
                if self._mode == "erg":
                    await self._apply_erg()
                else:
                    await self._apply_grade()
                self.connected.emit()
                self.status_changed.emit("Connected ● KICKR Core")
                while self._running:
                    await asyncio.sleep(1)
                await self._ftms.reset()
        except Exception as e:
            self.status_changed.emit(f"BLE Error: {e}")
            self.disconnected.emit()

    async def _disconnect(self):
        if self._ftms:
            try:
                await self._ftms.reset()
            except Exception:
                pass

    async def _apply_grade(self):
        if not self._ftms:
            return
        try:
            await self._ftms.set_target_incline(int(self._target_grade * 10))
        except Exception:
            try:
                await self._ftms.set_simulation_parameters(
                    0, int(self._target_grade * 100), 4, 9)
            except Exception:
                try:
                    w = max(50, min(1000, int(150 + self._target_grade * 10)))
                    await self._ftms.set_target_power(w)
                except Exception:
                    pass

    async def _apply_erg(self):
        if self._ftms and self._target_power:
            try:
                await self._ftms.set_target_power(self._target_power)
            except Exception:
                pass

    def _handle_data(self, data):
        self.data_received.emit(
            float(data.instant_cadence  or 0),
            float(data.instant_speed    or 0),
            float(data.total_distance   or 0),
            float(data.instant_power    or 0))


# ─────────────────────────────────────────────────────────────
#  GPX Helpers
# ─────────────────────────────────────────────────────────────

def load_gpx(path: str):
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                points.append((pt.latitude, pt.longitude, pt.elevation or 0.0))
    if not points:
        for wpt in gpx.waypoints:
            points.append((wpt.latitude, wpt.longitude, wpt.elevation or 0.0))
    return points


def compute_distances(points):
    dists = [0.0]
    for i in range(1, len(points)):
        lat1, lon1, _ = points[i-1]
        lat2, lon2, _ = points[i]
        dists.append(dists[-1] + haversine(lat1, lon1, lat2, lon2))
    return dists


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def grade_at_index(points, dists, idx, window=3):
    lo = max(0, idx - window)
    hi = min(len(points)-1, idx + window)
    if dists[hi] - dists[lo] < 1:
        return 0.0
    return ((points[hi][2] - points[lo][2]) / (dists[hi] - dists[lo])) * 100.0


# ─────────────────────────────────────────────────────────────
#  Elevation Profile Widget
# ─────────────────────────────────────────────────────────────

class ElevationProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._dists = []; self._eles = []; self._pos_frac = 0.0

    def set_route(self, dists, eles):
        self._dists = dists; self._eles = eles; self._pos_frac = 0.0; self.update()

    def set_progress(self, frac):
        self._pos_frac = max(0.0, min(1.0, frac)); self.update()

    def paintEvent(self, event):
        if not self._dists: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height(); pad = 8
        total = self._dists[-1]; mn, mx = min(self._eles), max(self._eles)
        span = mx - mn or 1

        def to_xy(i):
            x = pad + (self._dists[i] / total) * (W - 2*pad)
            y = (H - pad) - ((self._eles[i] - mn) / span) * (H - 2*pad)
            return x, y

        n = len(self._dists); pts = [to_xy(i) for i in range(n)]
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(255, 140, 0, 160))
        grad.setColorAt(1.0, QColor(255, 140, 0, 20))
        poly = QPolygonF()
        poly.append(QPointF(pts[0][0], H))
        for x, y in pts: poly.append(QPointF(x, y))
        poly.append(QPointF(pts[-1][0], H))
        p.setBrush(QBrush(grad)); p.setPen(Qt.PenStyle.NoPen); p.drawPolygon(poly)
        p.setPen(QPen(QColor("#FF8C00"), 2))
        for i in range(1, n):
            p.drawLine(int(pts[i-1][0]), int(pts[i-1][1]), int(pts[i][0]), int(pts[i][1]))
        prog_x = pad + self._pos_frac * (W - 2*pad)
        p.setPen(QPen(QColor("#00FF88"), 2, Qt.PenStyle.DashLine))
        p.drawLine(int(prog_x), 0, int(prog_x), H)
        p.setBrush(QBrush(QColor(0, 255, 136, 30))); p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(pad, 0, int(prog_x - pad), H); p.end()


# ─────────────────────────────────────────────────────────────
#  HR Graph Widget
# ─────────────────────────────────────────────────────────────

class HRGraph(QWidget):
    """Scrolling heart-rate graph."""
    MAX_POINTS = 300

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values: List[int] = []
        self._max_hr = 200

    def add_point(self, bpm: int, max_hr: int = 200):
        self._max_hr = max_hr
        self._values.append(bpm)
        if len(self._values) > self.MAX_POINTS:
            self._values.pop(0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height(); pad = 4
        p.fillRect(0, 0, W, H, QColor(20, 20, 20))

        if len(self._values) < 2:
            p.end(); return

        # Zone bands
        for zi, (lo, hi) in enumerate(HR_ZONE_BOUNDS):
            y1 = H - pad - int(hi * (H - 2*pad))
            y2 = H - pad - int(lo * (H - 2*pad))
            p.fillRect(0, max(0, y1), W, min(H, y2 - y1),
                       QColor(HR_ZONE_COLORS[zi]).darker(300))

        # HR line
        n = len(self._values)
        pts = []
        for i, v in enumerate(self._values):
            x = pad + (i / (n - 1)) * (W - 2*pad)
            y = (H - pad) - (v / self._max_hr) * (H - 2*pad)
            pts.append((x, y))

        poly = QPolygonF()
        for x, y in pts: poly.append(QPointF(x, y))
        p.setPen(QPen(QColor("#FF4466"), 2)); p.drawPolyline(poly)
        p.end()


# ─────────────────────────────────────────────────────────────
#  ZWO Workout Profile Widget
# ─────────────────────────────────────────────────────────────

class WorkoutProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._segments: List[ZwoSegment] = []
        self._total_s  = 0; self._pos_frac = 0.0; self._profile = None

    def set_workout(self, segments, profile):
        self._segments = segments; self._total_s = total_zwo_duration(segments)
        self._profile = profile; self._pos_frac = 0.0; self.update()

    def set_progress(self, frac):
        self._pos_frac = max(0.0, min(1.0, frac)); self.update()

    def paintEvent(self, event):
        if not self._segments or self._total_s == 0: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height(); pad = 6; usable = W - 2*pad
        max_pwr = min(max(
            max(s.power_frac, s.power_frac_end if s.is_ramp else s.power_frac)
            for s in self._segments) or 1.0, 1.5)
        t = 0
        for seg in self._segments:
            x = pad + (t / self._total_s) * usable
            w = max(1, int((seg.duration_s / self._total_s) * usable))
            plo = seg.power_frac; phi = seg.power_frac_end if seg.is_ramp else seg.power_frac
            h_lo = max(4, int((plo / max_pwr) * (H - 2*pad)))
            h_hi = max(4, int((phi / max_pwr) * (H - 2*pad)))
            mid_pwr = (plo + phi) / 2
            zone_idx, _ = self._profile.zone_for_pct(mid_pwr)
            color = QColor(ZONE_COLORS[min(zone_idx - 1, 6)])
            poly = QPolygonF([
                QPointF(x,     H - pad),     QPointF(x + w, H - pad),
                QPointF(x + w, H - pad - h_hi), QPointF(x, H - pad - h_lo)])
            p.setBrush(QBrush(color)); p.setPen(Qt.PenStyle.NoPen); p.drawPolygon(poly)
            t += seg.duration_s
        prog_x = pad + self._pos_frac * usable
        p.setPen(QPen(QColor("#FFFFFF"), 2))
        p.drawLine(int(prog_x), 0, int(prog_x), H); p.end()


# ─────────────────────────────────────────────────────────────
#  Stat Card
# ─────────────────────────────────────────────────────────────

class StatCard(QFrame):
    def __init__(self, label, unit, color="#FF8C00"):
        super().__init__(); self._color = color
        self.setObjectName("statCard")
        self.setStyleSheet("""#statCard { background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }""")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(1)
        self._val_lbl  = QLabel("—")
        self._val_lbl.setFont(QFont("Inter", 22, QFont.Weight.Bold))
        self._val_lbl.setStyleSheet(f"color: {color}; background: transparent;")
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setFont(QFont("Inter", 9))
        self._unit_lbl.setStyleSheet("color: rgba(255,255,255,0.45); background: transparent;")
        self._unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl = QLabel(label.upper())
        self._name_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self._name_lbl.setStyleSheet("color: rgba(255,255,255,0.35); background: transparent;")
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._name_lbl)
        layout.addWidget(self._val_lbl)
        layout.addWidget(self._unit_lbl)

    def set_value(self, v, decimals=0):
        self._val_lbl.setText(f"{v:.{decimals}f}" if isinstance(v, float) else str(v))

    def set_color(self, color: str):
        self._val_lbl.setStyleSheet(f"color: {color}; background: transparent;")


# ─────────────────────────────────────────────────────────────
#  HR Device Scanner Dialog
# ─────────────────────────────────────────────────────────────

class HRScanDialog(QDialog):
    device_selected = pyqtSignal(str, str)   # name, address

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Heart Rate Device")
        self.setMinimumSize(460, 380)
        self.setStyleSheet("""
            QDialog, QWidget { background:#1e1e1e; color:#fff;
                               font-family:'Inter',monospace; }
            QListWidget { background:#111; border:1px solid #333;
                          border-radius:6px; color:#fff; font-size:12px; }
            QListWidget::item:selected { background:#FF8C00; color:#000; }
            QPushButton { background:rgba(255,255,255,0.07);
                border:1px solid rgba(255,255,255,0.15); border-radius:6px;
                padding:6px 14px; color:#fff; font-size:11px; font-weight:bold; }
            QPushButton:hover { background:rgba(255,140,0,0.25); border-color:#FF8C00; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(10)

        hdr = QLabel("🫀  Heart Rate Monitor")
        hdr.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        hdr.setStyleSheet("color:#FF4466;")
        layout.addWidget(hdr)

        self._status = QLabel("Press Scan to discover BLE devices…")
        self._status.setStyleSheet("color:rgba(255,255,255,0.5); font-size:11px;")
        layout.addWidget(self._status)

        # Manual entry
        manual_row = QHBoxLayout()
        self._manual_edit = QLineEdit()
        self._manual_edit.setPlaceholderText("Or enter MAC address manually…")
        self._manual_edit.setStyleSheet(
            "background:#111; border:1px solid #333; border-radius:4px;"
            "padding:5px; color:#fff; font-size:11px;")
        manual_row.addWidget(self._manual_edit)
        btn_manual = QPushButton("Use")
        btn_manual.setFixedWidth(60)
        btn_manual.clicked.connect(self._use_manual)
        manual_row.addWidget(btn_manual)
        layout.addLayout(manual_row)

        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_scan   = QPushButton("🔍  Scan")
        self._btn_use    = QPushButton("✔  Connect")
        self._btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self._btn_scan)
        btn_row.addWidget(self._btn_use)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        self._btn_scan.clicked.connect(self._do_scan)
        self._btn_use.clicked.connect(self._use_selected)
        self._btn_cancel.clicked.connect(self.reject)

        # Preferred device
        self._list.addItem(QListWidgetItem(
            f"⭐  Garmin Vivoactive 5  [{GARMIN_HR_MAC}]"))
        self._list.item(0).setData(Qt.ItemDataRole.UserRole, GARMIN_HR_MAC)

        self._hr_worker = HRWorker()
        self._hr_worker.scan_result.connect(self._on_scan_result)
        self._hr_worker.status_changed.connect(
            lambda s: self._status.setText(s))

    def _do_scan(self):
        self._btn_scan.setEnabled(False)
        self._status.setText("Scanning…")
        self._hr_worker.scan()
        QTimer.singleShot(10000, lambda: self._btn_scan.setEnabled(True))

    @pyqtSlot(list)
    def _on_scan_result(self, devices):
        self._list.clear()
        self._list.addItem(QListWidgetItem(
            f"⭐  Garmin Vivoactive 5  [{GARMIN_HR_MAC}]"))
        self._list.item(0).setData(Qt.ItemDataRole.UserRole, GARMIN_HR_MAC)
        for name, addr in devices:
            item = QListWidgetItem(f"  {name}  [{addr}]")
            item.setData(Qt.ItemDataRole.UserRole, addr)
            self._list.addItem(item)
        self._btn_scan.setEnabled(True)

    def _use_selected(self):
        items = self._list.selectedItems()
        if not items:
            return
        addr = items[0].data(Qt.ItemDataRole.UserRole)
        name = items[0].text()
        self.device_selected.emit(name, addr)
        self.accept()

    def _use_manual(self):
        addr = self._manual_edit.text().strip()
        if addr:
            self.device_selected.emit("Manual Device", addr)
            self.accept()


# ─────────────────────────────────────────────────────────────
#  Profile Dialog
# ─────────────────────────────────────────────────────────────

class ProfileDialog(QDialog):
    def __init__(self, profile: CyclistProfile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cyclist Profile")
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog, QWidget { background:#1e1e1e; color:#fff;
                               font-family:'Inter',monospace; }
            QGroupBox { border:1px solid rgba(255,255,255,0.12); border-radius:8px;
                        margin-top:8px; padding:10px;
                        color:rgba(255,255,255,0.5); font-size:10px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
            QLabel { color:rgba(255,255,255,0.7); font-size:12px; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15);
                border-radius:6px; padding:4px 8px; color:#fff; font-size:12px; min-height:28px; }
            QDialogButtonBox QPushButton {
                background:rgba(255,140,0,0.2); border:1px solid #FF8C00;
                border-radius:6px; padding:6px 18px; color:#FF8C00; font-weight:bold; }
            QDialogButtonBox QPushButton:hover { background:rgba(255,140,0,0.4); }
        """)
        layout = QVBoxLayout(self); layout.setSpacing(14); layout.setContentsMargins(20,20,20,20)
        hdr = QLabel("⚙  Cyclist Profile"); hdr.setFont(QFont("Inter",14,QFont.Weight.Bold))
        hdr.setStyleSheet("color:#FF8C00;"); layout.addWidget(hdr)
        sub = QLabel("Your data improves power targets and calorie estimates.")
        sub.setStyleSheet("color:rgba(255,255,255,0.4); font-size:11px;"); sub.setWordWrap(True)
        layout.addWidget(sub)

        body_group = QGroupBox("Body Metrics"); body_form = QFormLayout(body_group)
        body_form.setSpacing(10)
        self._height = QDoubleSpinBox(); self._height.setRange(100,230)
        self._height.setSuffix(" cm"); self._height.setValue(profile.height_cm)
        self._weight = QDoubleSpinBox(); self._weight.setRange(30,250)
        self._weight.setSuffix(" kg"); self._weight.setDecimals(1)
        self._weight.setValue(profile.weight_kg)
        self._age = QSpinBox(); self._age.setRange(10,100)
        self._age.setSuffix(" yrs"); self._age.setValue(profile.age)
        self._gender = QComboBox(); self._gender.addItems(["Male","Female","Other"])
        self._gender.setCurrentText(profile.gender)
        body_form.addRow("Height", self._height); body_form.addRow("Weight", self._weight)
        body_form.addRow("Age", self._age); body_form.addRow("Gender", self._gender)
        layout.addWidget(body_group)

        train_group = QGroupBox("Training Data"); train_form = QFormLayout(train_group)
        train_form.setSpacing(10)
        self._ftp = QSpinBox(); self._ftp.setRange(50,600)
        self._ftp.setSuffix(" W"); self._ftp.setValue(profile.ftp_watts)
        ftp_hint = QLabel("FTP = your best average power over 60 min.")
        ftp_hint.setStyleSheet("color:rgba(255,255,255,0.3); font-size:10px;")
        ftp_hint.setWordWrap(True)
        train_form.addRow("FTP", self._ftp); train_form.addRow("", ftp_hint)
        layout.addWidget(train_group)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_profile(self) -> CyclistProfile:
        return CyclistProfile(
            height_cm=self._height.value(), weight_kg=self._weight.value(),
            ftp_watts=self._ftp.value(), age=self._age.value(),
            gender=self._gender.currentText())


# ─────────────────────────────────────────────────────────────
#  Workout History Panel
# ─────────────────────────────────────────────────────────────

class HistoryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[HistoryEntry] = []
        layout = QVBoxLayout(self); layout.setContentsMargins(8,8,8,8); layout.setSpacing(8)

        hdr = QLabel("WORKOUT HISTORY")
        hdr.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        hdr.setStyleSheet("color:#FF8C00;")
        layout.addWidget(hdr)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Date", "Distance", "Duration", "Avg Power", "Avg HR",
             "Calories", "Route", "ZWO"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet("""
            QTableWidget { background:#111; color:#fff; border:1px solid #333;
                           gridline-color:#222; font-size:11px; }
            QHeaderView::section { background:#1e1e1e; color:#888; border:none;
                                   padding:5px; font-size:10px; font-weight:bold; }
            QTableWidget::item:selected { background:rgba(255,140,0,0.2); }
        """)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_view_fit     = QPushButton("📄  View FIT")
        self._btn_view_route   = QPushButton("🗺  Route Image")
        self._btn_view_summary = QPushButton("📊  Summary Image")
        self._btn_export       = QPushButton("💾  Export FIT")
        self._btn_delete       = QPushButton("🗑  Delete")
        for btn in [self._btn_view_fit, self._btn_view_route,
                    self._btn_view_summary, self._btn_export, self._btn_delete]:
            btn.setFixedHeight(34)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self._btn_view_fit.clicked.connect(self._view_fit_info)
        self._btn_view_route.clicked.connect(self._view_route_img)
        self._btn_view_summary.clicked.connect(self._view_summary_img)
        self._btn_export.clicked.connect(self._export_fit)
        self._btn_delete.clicked.connect(self._delete_entry)

        self.refresh()

    def refresh(self):
        self._entries = load_history()
        self._entries.sort(key=lambda e: e.date_str, reverse=True)
        self._table.setRowCount(0)
        for e in self._entries:
            row = self._table.rowCount(); self._table.insertRow(row)
            dur_s = e.duration_s
            dur_str = f"{dur_s//3600:02d}:{(dur_s%3600)//60:02d}:{dur_s%60:02d}"
            for col, val in enumerate([
                e.date_str,
                f"{e.distance_km:.2f} km",
                dur_str,
                f"{e.avg_power:.0f} W",
                f"{e.avg_hr:.0f} bpm",
                f"{e.calories:.0f} kcal",
                os.path.basename(e.gpx_name) if e.gpx_name else "—",
                e.zwo_name or "—",
            ]):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, item)

    def _selected_entry(self) -> Optional[HistoryEntry]:
        rows = self._table.selectedItems()
        if not rows: return None
        return self._entries[self._table.currentRow()]

    def _view_fit_info(self):
        e = self._selected_entry()
        if not e: return
        if os.path.exists(e.fit_path):
            info = (f"FIT File: {e.fit_path}\n"
                    f"Date: {e.date_str}\n"
                    f"Distance: {e.distance_km:.2f} km\n"
                    f"Duration: {e.duration_s//60} min\n"
                    f"Avg Power: {e.avg_power:.0f} W\n"
                    f"Avg HR: {e.avg_hr:.0f} bpm\n"
                    f"Calories: {e.calories:.0f} kcal")
            QMessageBox.information(self, "FIT File Info", info)
        else:
            QMessageBox.warning(self, "Not Found", f"FIT file not found:\n{e.fit_path}")

    def _view_route_img(self):
        e = self._selected_entry()
        if not e: return
        if os.path.exists(e.route_img):
            dlg = ImageViewDialog(e.route_img, "Route", self)
            dlg.exec()
        else:
            QMessageBox.warning(self, "Not Found", "Route image not found.")

    def _view_summary_img(self):
        e = self._selected_entry()
        if not e: return
        if os.path.exists(e.summary_img):
            dlg = ImageViewDialog(e.summary_img, "Summary", self)
            dlg.exec()
        else:
            QMessageBox.warning(self, "Not Found", "Summary image not found.")

    def _export_fit(self):
        e = self._selected_entry()
        if not e: return
        if not os.path.exists(e.fit_path):
            QMessageBox.warning(self, "Not Found", "FIT file not found."); return
        dest, _ = QFileDialog.getSaveFileName(self, "Export FIT", e.fit_path,
                                              "FIT Files (*.fit)")
        if dest:
            import shutil; shutil.copy2(e.fit_path, dest)
            QMessageBox.information(self, "Exported", f"Saved to:\n{dest}")

    def _delete_entry(self):
        e = self._selected_entry()
        if not e: return
        reply = QMessageBox.question(self, "Delete Workout",
            f"Delete workout from {e.date_str}?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: return
        for path in [e.fit_path, e.route_img, e.summary_img]:
            if path and os.path.exists(path):
                try: os.unlink(path)
                except Exception: pass
        self._entries = [x for x in self._entries if x is not e]
        save_history(self._entries); self.refresh()


class ImageViewDialog(QDialog):
    def __init__(self, path: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title); self.resize(900, 560)
        self.setStyleSheet("QDialog { background:#111; }")
        layout = QVBoxLayout(self); layout.setContentsMargins(4,4,4,4)
        lbl = QLabel()
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(880, 540, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(pix); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        close.setFixedHeight(32); layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


# ─────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPX Trainer by Barry Prinsloo")
        self.resize(1400, 900)
        self._apply_dark_theme()
        os.makedirs(WORKOUTS_DIR, exist_ok=True)

        # ── Profile ───────────────────────────────────────────
        self._profile = CyclistProfile.load()

        # ── GPX state ─────────────────────────────────────────
        self._gpx_points: List[Tuple] = []
        self._gpx_dists:  List[float] = []
        self._gpx_name    = ""
        self._route_total = 0.0
        self._total_dist  = 0.0
        self._current_idx = 0
        self._is_riding   = False
        self._kickr_connected = False
        self._map_tmp     = None

        # ── ZWO state ─────────────────────────────────────────
        self._zwo_segments: List[ZwoSegment] = []
        self._zwo_name      = ""
        self._zwo_total_s   = 0
        self._zwo_elapsed_s = 0.0
        self._zwo_active    = False

        # ── HR state ──────────────────────────────────────────
        self._hr_connected   = False
        self._current_hr     = 0
        self._hr_min         = 999
        self._hr_max         = 0
        self._hr_sum         = 0
        self._hr_count       = 0
        self._hr_zone_time   = [0] * 5   # seconds per HR zone

        # ── Calorie / power accumulators ─────────────────────
        self._total_calories = 0.0
        self._last_power_w   = 0.0
        self._last_speed     = 0.0
        self._last_cadence   = 0
        self._last_grade     = 0.0
        self._power_history: List[float] = []   # for NP computation

        # ── Ride timer ────────────────────────────────────────
        self._ride_start_time: Optional[datetime] = None
        self._elapsed_s       = 0

        # ── Recording ─────────────────────────────────────────
        self._recording      = False
        self._workout_records: List[WorkoutRecord] = []

        # ── Workers ───────────────────────────────────────────
        self._ble_worker = KICKRWorker()
        self._ble_worker.data_received.connect(self._on_trainer_data)
        self._ble_worker.status_changed.connect(self._on_ble_status)
        self._ble_worker.connected.connect(self._on_ble_connected)
        self._ble_worker.disconnected.connect(self._on_ble_disconnected)

        self._hr_worker = HRWorker()
        self._hr_worker.hr_received.connect(self._on_hr_data)
        self._hr_worker.status_changed.connect(self._on_hr_status)
        self._hr_worker.connected.connect(self._on_hr_connected)
        self._hr_worker.disconnected.connect(self._on_hr_disconnected)

        self._build_ui()

        # ── Timers ────────────────────────────────────────────
        self._map_timer = QTimer(); self._map_timer.setInterval(3000)
        self._map_timer.timeout.connect(self._refresh_map)

        self._tick_timer = QTimer(); self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._on_tick)

        # ── Auto-connect HR at startup ─────────────────────────
        QTimer.singleShot(1500, self._auto_connect_hr)

    # ─────────────────────────────────────────────────────────
    #  Theme
    # ─────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#141414; color:#fff;
                                   font-family:'Inter',monospace; }
            QPushButton { background:rgba(255,255,255,0.07); color:#fff;
                border:1px solid rgba(255,255,255,0.15); border-radius:8px;
                padding:6px 14px; font-size:11px; font-weight:bold; }
            QPushButton:hover { background:rgba(255,140,0,0.25); border-color:#FF8C00; }
            QPushButton:pressed { background:rgba(255,140,0,0.4); }
            QPushButton:disabled { color:rgba(255,255,255,0.25);
                                   border-color:rgba(255,255,255,0.07); }
            QLabel { background:transparent; }
            QTabWidget::pane { border:1px solid #333; background:#141414; }
            QTabBar::tab { background:#1e1e1e; color:#888; padding:8px 18px;
                           border:1px solid #333; border-bottom:none; }
            QTabBar::tab:selected { background:#141414; color:#FF8C00; }
        """)

    # ─────────────────────────────────────────────────────────
    #  UI Construction
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8); root_layout.setSpacing(6)

        # ── Tab bar ──────────────────────────────────────────
        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs)

        # ── TAB 1: Ride ──────────────────────────────────────
        ride_widget = QWidget()
        ride_layout = QVBoxLayout(ride_widget)
        ride_layout.setContentsMargins(8, 8, 8, 8); ride_layout.setSpacing(6)

        # Top bar
        top = QHBoxLayout(); top.setSpacing(8)
        title = QLabel("GPX TRAINER")
        title.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:#FF8C00;")
        self._btn_profile = QPushButton("👤 Profile")
        self._btn_profile.setFixedHeight(30); self._btn_profile.setFixedWidth(110)
        self._btn_profile.clicked.connect(self._open_profile_dialog)
        self._elapsed_lbl = QLabel("00:00:00")
        self._elapsed_lbl.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        self._elapsed_lbl.setStyleSheet("color:#00FF88;")
        self._status_lbl = QLabel("Not Connected")
        self._status_lbl.setFont(QFont("Inter", 10))
        self._status_lbl.setStyleSheet("color:rgba(255,255,255,0.45);")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(title); top.addWidget(self._btn_profile)
        top.addWidget(self._elapsed_lbl); top.addStretch(); top.addWidget(self._status_lbl)
        ride_layout.addLayout(top)

        # Main split: map | right panel
        main_split = QHBoxLayout(); main_split.setSpacing(8)

        # Map
        self._map_view = QWebEngineView()
        self._map_view.setMinimumWidth(580)
        self._map_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        settings = self._map_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self._show_placeholder_map()
        main_split.addWidget(self._map_view, 3)

        # Right panel
        right = QVBoxLayout(); right.setSpacing(6)

        # ── Stat cards: 5 rows × 2 ──────────────────────────
        self._card_power   = StatCard("Power",    "WATTS",  "#FF8C00")
        self._card_cadence = StatCard("Cadence",  "RPM",    "#00CFFF")
        self._card_speed   = StatCard("Speed",    "KM/H",   "#FFFFFF")
        self._card_grade   = StatCard("Grade",    "%",      "#FF4466")
        self._card_dist    = StatCard("Distance", "KM",     "#AAFFAA")
        self._card_remain  = StatCard("Remaining","KM",     "#FFDDAA")
        self._card_zone    = StatCard("Pwr Zone", "",       "#9b59b6")
        self._card_cal     = StatCard("Calories", "KCAL",   "#FF6B6B")
        self._card_hr      = StatCard("Heart Rate","BPM",   "#FF4466")
        self._card_hr_zone = StatCard("HR Zone",  "",       "#e03535")

        for row in [
            [self._card_power,  self._card_cadence],
            [self._card_speed,  self._card_grade],
            [self._card_dist,   self._card_remain],
            [self._card_zone,   self._card_cal],
            [self._card_hr,     self._card_hr_zone],
        ]:
            g = QHBoxLayout(); g.setSpacing(6)
            for w in row: g.addWidget(w)
            right.addLayout(g)

        # Progress bars
        for attr, label, style in [
            ("_progress",     "ROUTE PROGRESS",
             "stop:0 #FF8C00, stop:1 #FF4466"),
            ("_zwo_progress", "WORKOUT PROGRESS  —  no .zwo loaded",
             "stop:0 #9b59b6, stop:1 #00CFFF"),
        ]:
            lbl_w = QLabel(label)
            lbl_w.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            lbl_w.setStyleSheet("color:rgba(255,255,255,0.35);")
            if attr == "_zwo_progress":
                self._zwo_label = lbl_w
            bar = QProgressBar(); bar.setRange(0, 1000); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(6)
            bar.setStyleSheet(f"""
                QProgressBar {{ background:rgba(255,255,255,0.08);
                    border-radius:3px; border:none; }}
                QProgressBar::chunk {{ background:qlineargradient(
                    x1:0,y1:0,x2:1,y2:0,{style}); border-radius:3px; }}
            """)
            setattr(self, attr, bar)
            right.addWidget(lbl_w); right.addWidget(bar)

        # Buttons
        self._btn_load     = QPushButton("📂  Load GPX Route")
        self._btn_load_zwo = QPushButton("💪  Load ZWO Workout (optional)")
        self._btn_hr       = QPushButton("🫀  Connect Heart Rate Monitor")
        self._btn_connect  = QPushButton("🔵  Connect KICKR")
        self._btn_ride     = QPushButton("▶  Start Ride")
        self._btn_stop     = QPushButton("■  Stop")

        self._btn_ride.setEnabled(False); self._btn_stop.setEnabled(False)

        for btn in [self._btn_load, self._btn_load_zwo, self._btn_hr,
                    self._btn_connect, self._btn_ride, self._btn_stop]:
            btn.setFixedHeight(36); right.addWidget(btn)

        self._btn_load.clicked.connect(self._load_gpx)
        self._btn_load_zwo.clicked.connect(self._load_zwo)
        self._btn_hr.clicked.connect(self._open_hr_dialog)
        self._btn_connect.clicked.connect(self._connect_kickr)
        self._btn_ride.clicked.connect(self._start_ride)
        self._btn_stop.clicked.connect(self._stop_ride)

        self._profile_summary = QLabel(self._fmt_profile_summary())
        self._profile_summary.setStyleSheet(
            "color:rgba(255,255,255,0.3); font-size:9px; padding-top:2px;")
        self._profile_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self._profile_summary)
        right.addStretch()

        main_split.addLayout(right, 1)
        ride_layout.addLayout(main_split, 3)

        # Bottom panels (elevation + workout + HR graph)
        bottom = QHBoxLayout(); bottom.setSpacing(10)

        elev_col = QVBoxLayout()
        elev_lbl = QLabel("ELEVATION PROFILE")
        elev_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        elev_lbl.setStyleSheet("color:rgba(255,255,255,0.3);")
        self._elev_widget = ElevationProfile()
        elev_col.addWidget(elev_lbl); elev_col.addWidget(self._elev_widget)
        bottom.addLayout(elev_col, 1)

        wo_col = QVBoxLayout()
        self._workout_title_lbl = QLabel("WORKOUT PROFILE  —  no .zwo loaded")
        self._workout_title_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        self._workout_title_lbl.setStyleSheet("color:rgba(255,255,255,0.3);")
        self._workout_widget = WorkoutProfile()
        wo_col.addWidget(self._workout_title_lbl); wo_col.addWidget(self._workout_widget)
        bottom.addLayout(wo_col, 1)

        hr_col = QVBoxLayout()
        hr_lbl = QLabel("HEART RATE")
        hr_lbl.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        hr_lbl.setStyleSheet("color:rgba(255,255,255,0.3);")
        self._hr_graph = HRGraph()
        hr_col.addWidget(hr_lbl); hr_col.addWidget(self._hr_graph)
        bottom.addLayout(hr_col, 1)

        ride_layout.addLayout(bottom)
        self._tabs.addTab(ride_widget, "🚴  Ride")

        # ── TAB 2: History ───────────────────────────────────
        self._history_panel = HistoryPanel()
        self._tabs.addTab(self._history_panel, "📋  History")
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────
    def _fmt_profile_summary(self) -> str:
        p = self._profile
        return (f"👤 {p.weight_kg:.0f} kg  |  FTP {p.ftp_watts} W  |"
                f"  {p.age} yrs  |  {p.gender}  |  MaxHR ~{p.max_hr()}")

    def _show_placeholder_map(self):
        self._map_view.setHtml("""<!DOCTYPE html><html><body style="margin:0;
            background:#1a1a1a;display:flex;align-items:center;
            justify-content:center;height:100vh;color:rgba(255,255,255,0.2);
            font-family:'Inter';font-size:15px;">
            Load a GPX file to display the route map.</body></html>""")

    def _build_map(self, ridden_idx=0):
        if not self._gpx_points: return
        pts = self._gpx_points
        centre = (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
        m = folium.Map(location=centre, zoom_start=13, tiles="CartoDB dark_matter")
        coords = [(p[0], p[1]) for p in pts]
        folium.PolyLine(coords, color="#AD5353", weight=3, opacity=0.7).add_to(m)
        if ridden_idx > 0:
            folium.PolyLine(coords[:ridden_idx+1], color="#FF8C00",
                            weight=4, opacity=0.9).add_to(m)
        folium.CircleMarker(coords[0],  radius=8, color="#00FF88",
                            fill=True, fill_color="#00FF88", popup="Start").add_to(m)
        folium.CircleMarker(coords[-1], radius=8, color="#FF4466",
                            fill=True, fill_color="#FF4466", popup="Finish").add_to(m)
        if 0 < ridden_idx < len(coords):
            folium.CircleMarker(coords[ridden_idx], radius=10, color="#FFFFFF",
                                fill=True, fill_color="#FF8C00",
                                popup="Current position").add_to(m)
        if self._map_tmp is None:
            self._map_tmp = os.path.join(tempfile.gettempdir(), "kickr_map.html")
        m.save(self._map_tmp)
        self._map_view.load(QUrl.fromLocalFile(os.path.abspath(self._map_tmp)))

    # ─────────────────────────────────────────────────────────
    #  Auto-connect HR
    # ─────────────────────────────────────────────────────────
    def _auto_connect_hr(self):
        self._status_lbl.setText(f"Auto-connecting HR: {GARMIN_HR_MAC}…")
        self._hr_worker.set_target(GARMIN_HR_MAC)
        self._hr_worker.start()

    # ─────────────────────────────────────────────────────────
    #  Dialogs
    # ─────────────────────────────────────────────────────────
    def _open_profile_dialog(self):
        dlg = ProfileDialog(self._profile, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._profile = dlg.get_profile(); self._profile.save()
            self._profile_summary.setText(self._fmt_profile_summary())
            if self._zwo_segments:
                self._workout_widget.set_workout(self._zwo_segments, self._profile)

    def _open_hr_dialog(self):
        dlg = HRScanDialog(self)
        dlg.device_selected.connect(self._on_hr_device_selected)
        dlg.exec()

    @pyqtSlot(str, str)
    def _on_hr_device_selected(self, name: str, address: str):
        self._hr_worker.stop()
        self._hr_worker = HRWorker()
        self._hr_worker.hr_received.connect(self._on_hr_data)
        self._hr_worker.status_changed.connect(self._on_hr_status)
        self._hr_worker.connected.connect(self._on_hr_connected)
        self._hr_worker.disconnected.connect(self._on_hr_disconnected)
        self._hr_worker.set_target(address)
        self._hr_worker.start()
        self._status_lbl.setText(f"Connecting HR: {name}…")

    # ─────────────────────────────────────────────────────────
    #  GPX / ZWO loading
    # ─────────────────────────────────────────────────────────
    def _load_gpx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GPX Route", "", "GPX Files (*.gpx)")
        if not path: return
        try:
            self._gpx_points = load_gpx(path)
            if len(self._gpx_points) < 2:
                self._status_lbl.setText("GPX has too few points"); return
            self._gpx_name  = path
            self._gpx_dists = compute_distances(self._gpx_points)
            self._route_total = self._gpx_dists[-1]
            eles = [p[2] for p in self._gpx_points]
            self._elev_widget.set_route(self._gpx_dists, eles)
            self._build_map(0)
            km = self._route_total / 1000
            gain = max(eles) - min(eles)
            self._status_lbl.setText(f"Route loaded  {km:.1f} km  |  Δ{gain:.0f} m")
            self._btn_ride.setEnabled(True)
            self._total_dist = 0.0; self._current_idx = 0
            self._progress.setValue(0)
            self._card_remain.set_value(km, 2)
        except Exception as e:
            self._status_lbl.setText(f"GPX error: {e}")

    def _load_zwo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ZWO Workout", "", "Zwift Workout Files (*.zwo)")
        if not path: return
        try:
            name, desc, segs = parse_zwo(path)
            if not segs:
                self._status_lbl.setText("ZWO file has no segments"); return
            self._zwo_segments  = segs; self._zwo_name = name
            self._zwo_total_s   = total_zwo_duration(segs); self._zwo_elapsed_s = 0.0
            self._workout_widget.set_workout(segs, self._profile)
            mins = self._zwo_total_s // 60
            self._zwo_label.setText(f"WORKOUT PROGRESS  —  {name}  ({mins} min)")
            self._workout_title_lbl.setText(f"WORKOUT PROFILE  —  {name}")
            self._zwo_progress.setValue(0)
            self._status_lbl.setText(f"ZWO: {name}  |  {mins} min  |  {len(segs)} segs")
            short = name[:26] + "…" if len(name) > 26 else name
            self._btn_load_zwo.setText(f"💪  {short}")
        except Exception as e:
            self._status_lbl.setText(f"ZWO error: {e}")

    # ─────────────────────────────────────────────────────────
    #  KICKR connection
    # ─────────────────────────────────────────────────────────
    def _connect_kickr(self):
        self._btn_connect.setEnabled(False)
        self._status_lbl.setText("Connecting to KICKR…")
        self._ble_worker.start()

    # ─────────────────────────────────────────────────────────
    #  Ride start / stop
    # ─────────────────────────────────────────────────────────
    def _start_ride(self):
        if not self._gpx_points: return
        self._is_riding      = True
        self._recording      = True
        self._total_dist     = 0.0; self._current_idx = 0
        self._total_calories = 0.0; self._elapsed_s   = 0
        self._zwo_elapsed_s  = 0.0
        self._zwo_active     = bool(self._zwo_segments)
        self._hr_min = 999; self._hr_max = 0; self._hr_sum = 0; self._hr_count = 0
        self._hr_zone_time   = [0] * 5
        self._power_history  = []
        self._workout_records = []
        self._ride_start_time = datetime.now(timezone.utc)

        self._btn_ride.setEnabled(False); self._btn_stop.setEnabled(True)
        self._map_timer.start(); self._tick_timer.start()

        if self._kickr_connected:
            if self._zwo_active:
                _, pf = segment_and_power_at(self._zwo_segments, 0)
                self._ble_worker.set_power_target(self._profile.target_watts(pf))
            else:
                g = grade_at_index(self._gpx_points, self._gpx_dists, 0)
                self._ble_worker.set_grade(g)

    def _stop_ride(self):
        was_recording = self._recording
        self._is_riding = False; self._zwo_active = False
        self._recording = False
        self._map_timer.stop(); self._tick_timer.stop()
        self._btn_ride.setEnabled(True); self._btn_stop.setEnabled(False)
        if self._kickr_connected: self._ble_worker.set_grade(0.0)
        if was_recording and self._workout_records:
            self._finalise_workout()

    # ─────────────────────────────────────────────────────────
    #  Finalise & save workout
    # ─────────────────────────────────────────────────────────
    def _finalise_workout(self):
        reply = QMessageBox.question(
            self, "Save Workout",
            "Save this workout (FIT file + images)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        summary = WorkoutSummary(
            start_time   = self._ride_start_time,
            end_time     = datetime.now(timezone.utc),
            gpx_name     = self._gpx_name,
            zwo_name     = self._zwo_name,
            weight_kg    = self._profile.weight_kg,
            calories     = self._total_calories,
            records      = self._workout_records,
        )
        summary.compute_from_records(self._profile.ftp_watts)

        # File paths
        date_tag = self._ride_start_time.strftime("%Y-%m-%d_%H%M%S")
        fit_path     = os.path.join(WORKOUTS_DIR, f"Workout_{date_tag}.fit")
        route_path   = os.path.join(WORKOUTS_DIR, f"Workout_{date_tag}_route.png")
        summary_path = os.path.join(WORKOUTS_DIR, f"Workout_{date_tag}_summary.png")

        # Write FIT
        try:
            fit_bytes = write_fit_file(summary)
            with open(fit_path, "wb") as f:
                f.write(fit_bytes)
        except Exception as e:
            print(f"FIT write error: {e}")
            fit_path = ""

        # Generate images
        try:
            generate_route_image(summary, route_path)
        except Exception as e:
            print(f"Route image error: {e}"); route_path = ""
        try:
            generate_summary_image(summary, summary_path)
        except Exception as e:
            print(f"Summary image error: {e}"); summary_path = ""

        # Update history
        entry = HistoryEntry(
            date_str    = self._ride_start_time.strftime("%Y-%m-%d %H:%M"),
            fit_path    = fit_path,
            route_img   = route_path,
            summary_img = summary_path,
            distance_km = summary.total_distance / 1000,
            duration_s  = summary.elapsed_s,
            avg_power   = summary.avg_power,
            avg_hr      = summary.avg_hr,
            calories    = summary.calories,
            gpx_name    = os.path.basename(self._gpx_name),
            zwo_name    = self._zwo_name,
        )
        entries = load_history()
        entries.append(entry)
        save_history(entries)
        self._history_panel.refresh()

        msg = (f"Workout saved!\n\n"
               f"Distance: {summary.total_distance/1000:.2f} km\n"
               f"Duration: {summary.elapsed_s//60} min\n"
               f"Avg Power: {summary.avg_power:.0f} W\n"
               f"TSS: {summary.tss:.0f}\n"
               f"NP: {summary.np_watts:.0f} W\n"
               f"Calories: {summary.calories:.0f} kcal\n\n"
               f"Files saved to:\n{WORKOUTS_DIR}")
        QMessageBox.information(self, "Workout Saved", msg)

    # ─────────────────────────────────────────────────────────
    #  1-second tick
    # ─────────────────────────────────────────────────────────
    def _on_tick(self):
        if not self._is_riding: return
        self._elapsed_s += 1
        h, rem = divmod(self._elapsed_s, 3600)
        m, s   = divmod(rem, 60)
        self._elapsed_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

        # Calories
        cal = self._profile.calories_per_second(self._last_power_w, self._current_hr)
        self._total_calories += cal
        self._card_cal.set_value(int(self._total_calories))

        # HR zone accumulation
        if self._current_hr > 0:
            zi, _ = self._profile.hr_zone(self._current_hr)
            self._hr_zone_time[min(zi-1, 4)] += 1
            self._hr_graph.add_point(self._current_hr, self._profile.max_hr())

        # ZWO progression
        if self._zwo_active:
            self._zwo_elapsed_s += 1.0
            if self._zwo_elapsed_s >= self._zwo_total_s:
                self._zwo_active = False; self._zwo_progress.setValue(1000)
                self._status_lbl.setText("🏁 Workout Complete!")
                if self._kickr_connected: self._ble_worker.set_grade(0.0)
            else:
                seg, pf = segment_and_power_at(self._zwo_segments, self._zwo_elapsed_s)
                if seg and self._kickr_connected:
                    self._ble_worker.set_power_target(self._profile.target_watts(pf))
                if seg:
                    zi, zn = self._profile.zone_for_pct(pf)
                    self._card_zone.set_value(f"Z{zi}")
                    self._card_zone.set_color(ZONE_COLORS[min(zi-1,6)])
                frac = self._zwo_elapsed_s / self._zwo_total_s
                self._zwo_progress.setValue(int(frac * 1000))
                self._workout_widget.set_progress(frac)

        # Record data point
        if self._recording:
            lat = lon = alt = 0.0
            if self._gpx_points and self._current_idx < len(self._gpx_points):
                pt  = self._gpx_points[self._current_idx]
                lat, lon, alt = pt[0], pt[1], pt[2]
            rec = WorkoutRecord(
                timestamp  = datetime.now(timezone.utc),
                lat        = lat, lon=lon, altitude=alt,
                distance   = self._total_dist,
                speed      = self._last_speed,
                heart_rate = self._current_hr,
                cadence    = self._last_cadence,
                power      = int(self._last_power_w),
                grade      = self._last_grade,
            )
            self._workout_records.append(rec)

    # ─────────────────────────────────────────────────────────
    #  Trainer data
    # ─────────────────────────────────────────────────────────
    @pyqtSlot(float, float, float, float)
    def _on_trainer_data(self, cadence, speed, dist_m, power):
        self._last_power_w  = power
        self._last_speed    = speed
        self._last_cadence  = int(cadence)
        self._power_history.append(power)

        delta_m = (speed / 3.6) * 1.0
        if self._is_riding and self._gpx_dists:
            self._total_dist += delta_m
            td = min(self._total_dist, self._route_total)
            lo, hi = 0, len(self._gpx_dists) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self._gpx_dists[mid] < td: lo = mid + 1
                else: hi = mid
            self._current_idx = lo
            if not self._zwo_active:
                g = grade_at_index(self._gpx_points, self._gpx_dists, lo)
                self._last_grade = g
                if self._kickr_connected: self._ble_worker.set_grade(g)
                self._card_grade.set_value(g, 1)
            frac = td / self._route_total
            self._progress.setValue(int(frac * 1000))
            self._elev_widget.set_progress(frac)
            self._card_dist.set_value(td / 1000, 2)
            self._card_remain.set_value((self._route_total - td) / 1000, 2)
            if self._total_dist >= self._route_total:
                self._stop_ride(); self._status_lbl.setText("🏁 Route Complete!")

        self._card_power.set_value(int(power))
        self._card_cadence.set_value(int(cadence))
        self._card_speed.set_value(speed, 1)

    # ─────────────────────────────────────────────────────────
    #  HR data
    # ─────────────────────────────────────────────────────────
    @pyqtSlot(int)
    def _on_hr_data(self, bpm: int):
        self._current_hr = bpm
        if bpm > 0:
            self._hr_max = max(self._hr_max, bpm)
            self._hr_min = min(self._hr_min, bpm)
            self._hr_sum += bpm; self._hr_count += 1
        zi, zn = self._profile.hr_zone(bpm)
        zc = HR_ZONE_COLORS[min(zi-1, 4)]
        self._card_hr.set_value(bpm)
        self._card_hr.set_color(zc)
        self._card_hr_zone.set_value(f"Z{zi}")
        self._card_hr_zone.set_color(zc)

    @pyqtSlot(str)
    def _on_hr_status(self, msg: str):
        # Only show HR status when not showing KICKR status
        if "KICKR" not in self._status_lbl.text():
            self._status_lbl.setText(msg)

    @pyqtSlot(str)
    def _on_hr_connected(self, addr: str):
        self._hr_connected = True
        self._btn_hr.setText(f"✔ HR Connected")
        self._btn_hr.setStyleSheet(
            self._btn_hr.styleSheet() + "border-color: #FF4466;")

    @pyqtSlot()
    def _on_hr_disconnected(self):
        self._hr_connected = False
        self._btn_hr.setText("🫀  Connect Heart Rate Monitor")
        # Attempt reconnection after 5 s
        QTimer.singleShot(5000, self._hr_worker.reconnect)

    # ─────────────────────────────────────────────────────────
    #  KICKR signals
    # ─────────────────────────────────────────────────────────
    @pyqtSlot(str)
    def _on_ble_status(self, msg): self._status_lbl.setText(msg)

    @pyqtSlot()
    def _on_ble_connected(self):
        self._kickr_connected = True
        self._btn_connect.setText("✔  KICKR Connected")

    @pyqtSlot()
    def _on_ble_disconnected(self):
        self._kickr_connected = False
        self._btn_connect.setEnabled(True)
        self._btn_connect.setText("🔵  Connect KICKR")

    def _refresh_map(self):
        if self._gpx_points and self._is_riding:
            self._build_map(self._current_idx)

    def _on_tab_changed(self, idx: int):
        if idx == 1:
            self._history_panel.refresh()

    def closeEvent(self, event):
        self._ble_worker.stop(); self._hr_worker.stop()
        if self._map_tmp and os.path.exists(self._map_tmp):
            try: os.unlink(self._map_tmp)
            except Exception: pass
        event.accept()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--disable-web-security --allow-file-access-from-files")
    app = QApplication(sys.argv)
    app.setApplicationName("KICKR GPX Trainer")
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
