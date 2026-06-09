"""
KICKR GPX Trainer App
---------------------
Loads a GPX file, displays the route on an interactive map,
and controls a Wahoo KICKR Core via BLE FTMS to simulate
the route's gradient in real-time.

Dependencies:
    pip install PyQt6 PyQtWebEngine bleak pycycling gpxpy folium numpy
"""

import sys
import os
import asyncio
import threading
import math
import json
import tempfile
import gpxpy
import folium
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QFrame, QSizePolicy,
    QProgressBar, QGraphicsDropShadowEffect
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, pyqtSlot
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush,
    QLinearGradient, QPixmap, QFontDatabase
)

# ─────────────────────────────────────────────────────────────
#  BLE Worker  (runs the asyncio event loop in a background thread)
# ─────────────────────────────────────────────────────────────

class KICKRWorker(QObject):
    """Runs bleak/pycycling in a dedicated asyncio thread."""
    data_received   = pyqtSignal(float, float, float, float)   # cadence, speed, dist, power
    status_changed  = pyqtSignal(str)
    connected       = pyqtSignal()
    disconnected    = pyqtSignal()

    KICKR_ADDRESS = "D2:9D:40:FB:E7:22"

    def __init__(self):
        super().__init__()
        self._loop     = None
        self._client   = None
        self._ftms     = None
        self._running  = False
        self._target_grade = 0.0   # % grade  -20 … +20
        self._grade_dirty  = False

    # ── called from Qt thread ──────────────────────────────
    def set_grade(self, grade_pct: float):
        self._target_grade = max(-20.0, min(20.0, grade_pct))
        self._grade_dirty  = True
        if self._loop and self._ftms:
            asyncio.run_coroutine_threadsafe(self._apply_grade(), self._loop)

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    # ── private asyncio coroutines ─────────────────────────
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run())

    async def _connect_and_run(self):
        from bleak import BleakClient
        from pycycling.fitness_machine_service import FitnessMachineService

        self.status_changed.emit(f"Connecting to KICKR…")
        try:
            async with BleakClient(self.KICKR_ADDRESS) as client:
                self._client = client
                self._ftms   = FitnessMachineService(client)
                self._ftms.set_indoor_bike_data_handler(self._handle_data)
                await self._ftms.request_control()
                await self._ftms.enable_indoor_bike_data_notify()
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
        if self._ftms is None:
            return
        try:
            # FTMS incline uses 0.1% resolution (signed int), so 5.0% → 50
            grade_int = int(self._target_grade * 10)
            await self._ftms.set_target_incline(grade_int)
        except Exception:
            # Fall back to simulation parameters (wind=0, grade in 0.01% units, crr, cw)
            try:
                grade_sim = int(self._target_grade * 100)  # 0.01% resolution
                await self._ftms.set_simulation_parameters(0, grade_sim, 4, 9)
            except Exception:
                # Last resort: ERG mode approximation
                watts = int(150 + self._target_grade * 10)
                watts = max(50, min(1000, watts))
                try:
                    await self._ftms.set_target_power(watts)
                except Exception:
                    pass

    def _handle_data(self, data):
        cadence = data.instant_cadence  or 0.0
        speed   = data.instant_speed    or 0.0
        dist    = data.total_distance   or 0.0
        power   = data.instant_power    or 0.0
        self.data_received.emit(float(cadence), float(speed), float(dist), float(power))


# ─────────────────────────────────────────────────────────────
#  GPX helpers
# ─────────────────────────────────────────────────────────────

def load_gpx(path: str):
    """Return list of (lat, lon, ele_m) tuples."""
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
    """Cumulative distance in metres along the route."""
    dists = [0.0]
    for i in range(1, len(points)):
        lat1, lon1, _ = points[i-1]
        lat2, lon2, _ = points[i]
        dists.append(dists[-1] + haversine(lat1, lon1, lat2, lon2))
    return dists


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def grade_at_index(points, dists, idx, window=3):
    """Smoothed gradient (%) at point idx."""
    lo = max(0, idx - window)
    hi = min(len(points)-1, idx + window)
    if dists[hi] - dists[lo] < 1:
        return 0.0
    rise = points[hi][2] - points[lo][2]
    run  = dists[hi] - dists[lo]
    return (rise / run) * 100.0


# ─────────────────────────────────────────────────────────────
#  Elevation Profile Widget  (custom QPainter)
# ─────────────────────────────────────────────────────────────

class ElevationProfile(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._dists   = []
        self._eles    = []
        self._pos_frac = 0.0   # 0..1 progress

    def set_route(self, dists, eles):
        self._dists = dists
        self._eles  = eles
        self._pos_frac = 0.0
        self.update()

    def set_progress(self, frac):
        self._pos_frac = max(0.0, min(1.0, frac))
        self.update()

    def paintEvent(self, event):
        if not self._dists:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        pad = 8

        total = self._dists[-1]
        mn, mx = min(self._eles), max(self._eles)
        span = mx - mn or 1

        def to_xy(i):
            x = pad + (self._dists[i] / total) * (W - 2*pad)
            y = (H - pad) - ((self._eles[i] - mn) / span) * (H - 2*pad)
            return x, y

        n = len(self._dists)
        pts = [to_xy(i) for i in range(n)]

        # gradient fill
        grad = QLinearGradient(0, 0, 0, H)
        grad.setColorAt(0.0, QColor(255, 140, 0, 160))
        grad.setColorAt(1.0, QColor(255, 140, 0, 20))
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtCore import QPointF
        poly = QPolygonF()
        poly.append(QPointF(pts[0][0], H))
        for x, y in pts:
            poly.append(QPointF(x, y))
        poly.append(QPointF(pts[-1][0], H))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(poly)

        # line
        p.setPen(QPen(QColor("#FF8C00"), 2))
        for i in range(1, n):
            p.drawLine(int(pts[i-1][0]), int(pts[i-1][1]),
                       int(pts[i][0]),   int(pts[i][1]))

        # progress highlight
        prog_x = pad + self._pos_frac * (W - 2*pad)
        p.setPen(QPen(QColor("#00FF88"), 2, Qt.PenStyle.DashLine))
        p.drawLine(int(prog_x), 0, int(prog_x), H)

        # ridden area tint
        p.setBrush(QBrush(QColor(0, 255, 136, 30)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(pad, 0, int(prog_x - pad), H)

        p.end()


# ─────────────────────────────────────────────────────────────
#  Stat Card Widget
# ─────────────────────────────────────────────────────────────

class StatCard(QFrame):
    def __init__(self, label, unit, color="#FF8C00"):
        super().__init__()
        self._color = color
        self.setObjectName("statCard")
        self.setStyleSheet(f"""
            #statCard {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        self._val_lbl = QLabel("—")
        self._val_lbl.setFont(QFont("Inter", 28, QFont.Weight.Bold))
        self._val_lbl.setStyleSheet(f"color: {color}; background: transparent;")
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setFont(QFont("Inter", 10))
        self._unit_lbl.setStyleSheet("color: rgba(255,255,255,0.45); background: transparent;")
        self._unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._name_lbl = QLabel(label.upper())
        self._name_lbl.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        self._name_lbl.setStyleSheet("color: rgba(255,255,255,0.35); background: transparent;")
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._name_lbl)
        layout.addWidget(self._val_lbl)
        layout.addWidget(self._unit_lbl)

    def set_value(self, v, decimals=0):
        if isinstance(v, float):
            self._val_lbl.setText(f"{v:.{decimals}f}")
        else:
            self._val_lbl.setText(str(v))


# ─────────────────────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPX Trainer by Barry Prinsloo")
        self.resize(1200, 780)

        self._apply_dark_theme()

        # State
        self._gpx_points  = []
        self._gpx_dists   = []
        self._total_dist  = 0.0   # metres ridden (from trainer)
        self._route_total = 0.0   # metres total in GPX
        self._current_idx = 0
        self._is_riding   = False
        self._kickr_connected = False
        self._map_tmp     = None  # temp html path

        # BLE worker
        self._ble_worker = KICKRWorker()
        self._ble_worker.data_received.connect(self._on_trainer_data)
        self._ble_worker.status_changed.connect(self._on_ble_status)
        self._ble_worker.connected.connect(self._on_ble_connected)
        self._ble_worker.disconnected.connect(self._on_ble_disconnected)

        # UI
        self._build_ui()

        # Map update timer
        self._map_timer = QTimer()
        self._map_timer.setInterval(3000)   # refresh map every 3 s
        self._map_timer.timeout.connect(self._refresh_map)

    # ── Theme ─────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #141414;
                color: #FFFFFF;
                font-family: 'Inter', monospace;
            }
            QPushButton {
                background: rgba(255,255,255,0.07);
                color: #FFFFFF;
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 8px;
                padding: 8px 18px;
                font-family: 'Inter', monospace;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255,140,0,0.25);
                border-color: #FF8C00;
            }
            QPushButton:pressed {
                background: rgba(255,140,0,0.4);
            }
            QPushButton:disabled {
                color: rgba(255,255,255,0.25);
                border-color: rgba(255,255,255,0.07);
            }
            QLabel { background: transparent; }
        """)

    # ── UI Construction ───────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(10)

        # ── Top bar ───────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)

        title = QLabel("GPX TRAINER")
        title.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #FF8C00;")

        self._status_lbl = QLabel("Not Connected")
        self._status_lbl.setFont(QFont("Inter", 11))
        self._status_lbl.setStyleSheet("color: rgba(255,255,255,0.45);")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        top.addWidget(title)
        top.addStretch()
        top.addWidget(self._status_lbl)
        root_layout.addLayout(top)

        # ── Main area  (map | controls) ───────────────────
        main = QHBoxLayout()
        main.setSpacing(10)

        # Map
        self._map_view = QWebEngineView()
        self._map_view.setMinimumWidth(620)
        self._map_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Allow local HTML to load remote tile resources (Leaflet CDN etc.)
        settings = self._map_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self._show_placeholder_map()
        main.addWidget(self._map_view, 3)

        # Right panel
        right = QVBoxLayout()
        right.setSpacing(10)

        # Stat cards  (2 × 2 grid)
        self._card_power   = StatCard("Power",   "WATTS",  "#FF8C00")
        self._card_cadence = StatCard("Cadence", "RPM",    "#00CFFF")
        self._card_speed   = StatCard("Speed",   "KM/H",   "#FFFFFF")
        self._card_grade   = StatCard("Grade",   "%",      "#FF4466")
        self._card_dist    = StatCard("Distance","KM",     "#AAFFAA")
        self._card_remain  = StatCard("Remaining","KM",    "#FFDDAA")

        g1 = QHBoxLayout(); g1.setSpacing(8)
        g1.addWidget(self._card_power)
        g1.addWidget(self._card_cadence)
        g2 = QHBoxLayout(); g2.setSpacing(8)
        g2.addWidget(self._card_speed)
        g2.addWidget(self._card_grade)
        g3 = QHBoxLayout(); g3.setSpacing(8)
        g3.addWidget(self._card_dist)
        g3.addWidget(self._card_remain)

        right.addLayout(g1)
        right.addLayout(g2)
        right.addLayout(g3)

        # Progress bar
        prog_lbl = QLabel("ROUTE PROGRESS")
        prog_lbl.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        prog_lbl.setStyleSheet("color: rgba(255,255,255,0.35);")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet("""
            QProgressBar { background: rgba(255,255,255,0.08); border-radius: 4px; border: none; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #FF8C00, stop:1 #FF4466); border-radius: 4px; }
        """)
        right.addWidget(prog_lbl)
        right.addWidget(self._progress)

        # Buttons
        self._btn_load    = QPushButton("📂  Load GPX Route")
        self._btn_connect = QPushButton("🔵  Connect KICKR")
        self._btn_ride    = QPushButton("▶   Start Ride")
        self._btn_stop    = QPushButton("■   Stop")
        self._btn_ride.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_ride.setStyleSheet(self._btn_ride.styleSheet() +
            "QPushButton { border-color: #00FF88; }")

        for btn in [self._btn_load, self._btn_connect,
                    self._btn_ride, self._btn_stop]:
            btn.setFixedHeight(40)
            right.addWidget(btn)

        self._btn_load.clicked.connect(self._load_gpx)
        self._btn_connect.clicked.connect(self._connect_kickr)
        self._btn_ride.clicked.connect(self._start_ride)
        self._btn_stop.clicked.connect(self._stop_ride)

        right.addStretch()
        main.addLayout(right, 1)
        root_layout.addLayout(main, 3)

        # ── Elevation profile ─────────────────────────────
        elev_lbl = QLabel("ELEVATION PROFILE")
        elev_lbl.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        elev_lbl.setStyleSheet("color: rgba(255,255,255,0.3);")
        root_layout.addWidget(elev_lbl)

        self._elev_widget = ElevationProfile()
        root_layout.addWidget(self._elev_widget)

    # ── Placeholder map ───────────────────────────────────
    def _show_placeholder_map(self):
        html = """<!DOCTYPE html><html><body style="margin:0;background:#1a1a1a;
        display:flex;align-items:center;justify-content:center;height:100vh;
        color:rgba(255,255,255,0.2);font-family:'Inter';font-size:16px;">
        Load a GPX file to display the route map.</body></html>"""
        self._map_view.setHtml(html)

    # ── Build folium map ──────────────────────────────────
    def _build_map(self, ridden_idx=0):
        if not self._gpx_points:
            return
        pts = self._gpx_points
        centre = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts)
        )
        m = folium.Map(location=centre, zoom_start=13,
                       tiles="CartoDB dark_matter")

        # Full route (grey)
        coords = [(p[0], p[1]) for p in pts]
        folium.PolyLine(coords, color="#AD5353", weight=3, opacity=0.7).add_to(m)

        # Ridden portion (orange)
        if ridden_idx > 0:
            ridden = coords[:ridden_idx+1]
            folium.PolyLine(ridden, color="#FF8C00", weight=4, opacity=0.9).add_to(m)

        # Start marker
        folium.CircleMarker(coords[0], radius=8, color="#00FF88",
                            fill=True, fill_color="#00FF88",
                            popup="Start").add_to(m)
        # End marker
        folium.CircleMarker(coords[-1], radius=8, color="#FF4466",
                            fill=True, fill_color="#FF4466",
                            popup="Finish").add_to(m)

        # Current position
        if ridden_idx > 0 and ridden_idx < len(coords):
            folium.CircleMarker(coords[ridden_idx], radius=10,
                                color="#FFFFFF", fill=True,
                                fill_color="#FF8C00",
                                popup="Current position").add_to(m)

        # Save to temp file – use a fixed path in the system temp dir
        if self._map_tmp is None:
            self._map_tmp = os.path.join(
                tempfile.gettempdir(), "kickr_trainer_map.html")
        m.save(self._map_tmp)
        from PyQt6.QtCore import QUrl
        # Force a reload by appending a cache-busting fragment
        url = QUrl.fromLocalFile(os.path.abspath(self._map_tmp))
        self._map_view.load(url)

    # ── Slots ─────────────────────────────────────────────
    def _load_gpx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GPX Route", "", "GPX Files (*.gpx)")
        if not path:
            return
        try:
            self._gpx_points = load_gpx(path)
            if len(self._gpx_points) < 2:
                self._status_lbl.setText("GPX has too few points")
                return
            self._gpx_dists  = compute_distances(self._gpx_points)
            self._route_total = self._gpx_dists[-1]
            eles = [p[2] for p in self._gpx_points]
            self._elev_widget.set_route(self._gpx_dists, eles)
            self._build_map(0)
            km = self._route_total / 1000
            gain = max(eles) - min(eles)
            self._status_lbl.setText(
                f"Route loaded  {km:.1f} km  |  Δ{gain:.0f} m")
            self._btn_ride.setEnabled(True)
            self._total_dist = 0.0
            self._current_idx = 0
            self._progress.setValue(0)
            self._card_remain.set_value(km, 2)
        except Exception as e:
            self._status_lbl.setText(f"GPX error: {e}")

    def _connect_kickr(self):
        self._btn_connect.setEnabled(False)
        self._status_lbl.setText("Connecting to KICKR…")
        self._ble_worker.start()

    def _start_ride(self):
        if not self._gpx_points:
            return
        self._is_riding  = True
        self._total_dist = 0.0
        self._current_idx = 0
        self._btn_ride.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._map_timer.start()
        # Push initial grade
        if self._kickr_connected:
            g = grade_at_index(self._gpx_points, self._gpx_dists, 0)
            self._ble_worker.set_grade(g)

    def _stop_ride(self):
        self._is_riding  = False
        self._map_timer.stop()
        self._btn_ride.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._kickr_connected:
            self._ble_worker.set_grade(0.0)

    @pyqtSlot(float, float, float, float)
    def _on_trainer_data(self, cadence, speed, dist_m, power):
        # Accumulate distance
        # dist_m from FTMS is cumulative total metres; use delta each call
        # (pycycling resets on reconnect so just track delta via speed)
        # We'll use speed (km/h) × time interval (assume ~1 s update) → metres
        delta_m = (speed / 3.6) * 1.0   # 1-second approximation
        self._total_dist += delta_m

        # Find current route point
        if self._is_riding and self._gpx_dists:
            # binary search for current index
            td = min(self._total_dist, self._route_total)
            lo, hi = 0, len(self._gpx_dists)-1
            while lo < hi:
                mid = (lo+hi)//2
                if self._gpx_dists[mid] < td:
                    lo = mid+1
                else:
                    hi = mid
            self._current_idx = lo

            # Grade at current point
            g = grade_at_index(self._gpx_points, self._gpx_dists, lo)
            self._ble_worker.set_grade(g)
            self._card_grade.set_value(g, 1)

            # Progress
            frac = td / self._route_total
            self._progress.setValue(int(frac * 1000))
            self._elev_widget.set_progress(frac)
            self._card_dist.set_value(td / 1000, 2)
            self._card_remain.set_value((self._route_total - td) / 1000, 2)

            # Auto-stop at end
            if self._total_dist >= self._route_total:
                self._stop_ride()
                self._status_lbl.setText("🏁 Ride Complete!")

        # Update stat cards
        self._card_power.set_value(int(power))
        self._card_cadence.set_value(int(cadence))
        self._card_speed.set_value(speed, 1)

    @pyqtSlot(str)
    def _on_ble_status(self, msg):
        self._status_lbl.setText(msg)

    @pyqtSlot()
    def _on_ble_connected(self):
        self._kickr_connected = True
        self._btn_connect.setText("✔  KICKR Connected")
        self._btn_connect.setStyleSheet(
            self._btn_connect.styleSheet() + "border-color: #00FF88;")

    @pyqtSlot()
    def _on_ble_disconnected(self):
        self._kickr_connected = False
        self._btn_connect.setEnabled(True)
        self._btn_connect.setText("🔵  Connect KICKR")

    def _refresh_map(self):
        if self._gpx_points and self._is_riding:
            self._build_map(self._current_idx)

    def closeEvent(self, event):
        self._ble_worker.stop()
        if self._map_tmp and os.path.exists(self._map_tmp):
            try:
                os.unlink(self._map_tmp)
            except Exception:
                pass
        event.accept()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    # Must be set BEFORE QApplication is created
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--disable-web-security --allow-file-access-from-files"
    )

    app = QApplication(sys.argv)
    app.setApplicationName("KICKR GPX Trainer")

    # Enable High-DPI
    from PyQt6.QtCore import Qt as _Qt
    try:
        app.setAttribute(_Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
