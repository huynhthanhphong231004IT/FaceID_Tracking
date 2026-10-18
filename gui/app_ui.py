from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, Dict, Any, List

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

import config
from gui import overlay
from models import ModelLoadError
from pipeline import (ST_ERROR, ST_FINISHED, ST_IDLE, ST_PAUSED, ST_RUNNING, FrameResult,
                      RecognitionPipeline)
from utils.tracker import STATE_KNOWN, STATE_UNKNOWN
from utils.video_source import VideoSourceError

logger = logging.getLogger(__name__)
C = config.COLORS

COLOR_THEME = {
    "bg_dark": "#02040A",       
    "card_bg": "#080D16",       
    "card_border": "#151F32",   
    "accent_cyan": "#00E5FF",   
    "accent_purple": "#A855F7", 
    "accent_green": "#10B981",  
    "accent_amber": "#F59E0B",  
    "accent_red": "#EF4444",    
    "text_main": "#F8FAFC",     
    "text_sub": "#64748B",      
    "panel_dark": "#040711",    
}

if sys.platform.startswith("win"):
    MONO = "Consolas"
elif sys.platform == "darwin":
    MONO = "Menlo"
else:
    MONO = "DejaVu Sans Mono"

CROP_SIZE = 168          


class StatCard(ctk.CTkFrame):
    def __init__(self, parent, title: str, value: str = "—", unit: str = "", color: str = COLOR_THEME["accent_cyan"]):
        super().__init__(parent, fg_color=COLOR_THEME["panel_dark"], corner_radius=10,
                         border_width=1, border_color=COLOR_THEME["card_border"])
        self.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self, text=title, font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"]
                     ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))
        
        val_box = ctk.CTkFrame(self, fg_color="transparent")
        val_box.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))
        
        self.val_label = ctk.CTkLabel(val_box, text=value, font=(MONO, 18, "bold"), text_color=color)
        self.val_label.pack(side="left")
        if unit:
            ctk.CTkLabel(val_box, text=f" {unit}", font=(MONO, 10), text_color=COLOR_THEME["text_sub"]
                         ).pack(side="left", pady=(4, 0))

    def update_val(self, value: str):
        self.val_label.configure(text=value)


class FaceTrackerApp(ctk.CTk):

    def __init__(self, pipeline: RecognitionPipeline) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        super().__init__()

        self.pipeline = pipeline
        self.title(config.APP_TITLE + " — Enterprise Edition")
        self.geometry("1520x900")
        self.minsize(1240, 768)
        self.configure(fg_color=COLOR_THEME["bg_dark"])

        self._ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._models_ready = False
        self._starting = False
        self._closing = False
        self._last_seq = -1
        self._last_state = ST_IDLE
        self._status_cache = ("", "")
        self._last_error_text = ""
        self._sx = 1.0
        self._sy = 1.0

        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._proc_time_ms = 0.0
        self._logs_cache: List[tuple] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_workspace()
        self._build_controls()
        self._set_buttons_state()
        
        self._set_status("INITIALIZING AI ENGINES", COLOR_THEME["accent_amber"])
        self._log("SYSTEM INIT: Loading ArcFace 512-D & YOLOv8-Face Neural Engines...", "info")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._load_models_bg, daemon=True, name="model-loader").start()
        self.after(config.UI_REFRESH_MS, self._tick)
    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=COLOR_THEME["card_bg"], corner_radius=0, height=64)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(2, weight=1)
        title_box = ctk.CTkFrame(bar, fg_color="transparent")
        title_box.grid(row=0, column=0, padx=(20, 12), pady=12)
        ctk.CTkLabel(
            title_box, text="FACE TRACKER", font=(MONO, 20, "bold"), text_color=COLOR_THEME["accent_cyan"]
        ).pack(side="left")
        ctk.CTkLabel(
            title_box, text=" ENTERPRISE ", font=(MONO, 10, "bold"), text_color=COLOR_THEME["accent_purple"],
            fg_color="#200B3B", corner_radius=4, padx=6, pady=2
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            bar, text="YOLOv8-Face  |  ArcFace 512-D  |  Real-Time Re-Identification System",
            font=(MONO, 11), text_color=COLOR_THEME["text_sub"]
        ).grid(row=0, column=1, sticky="w", padx=10)
        status_box = ctk.CTkFrame(bar, fg_color="transparent")
        status_box.grid(row=0, column=2, sticky="e", padx=20)
        self.fps_label = ctk.CTkLabel(
            status_box, text="FPS: --", font=(MONO, 11, "bold"),
            text_color=COLOR_THEME["accent_purple"], fg_color="#180B2B", corner_radius=6, padx=8, pady=3
        )
        self.fps_label.pack(side="left", padx=(0, 8))
        self.latency_label = ctk.CTkLabel(
            status_box, text="LATENCY: -- ms", font=(MONO, 11, "bold"),
            text_color=COLOR_THEME["accent_cyan"], fg_color="#032236", corner_radius=6, padx=8, pady=3
        )
        self.latency_label.pack(side="left", padx=(0, 12))
        self.status_label = ctk.CTkLabel(status_box, text="", font=(MONO, 12, "bold"), text_color=COLOR_THEME["accent_amber"])
        self.status_label.pack(side="left")
    def _build_workspace(self) -> None:
        self.tabview = ctk.CTkTabview(
            self, fg_color="transparent",
            segmented_button_fg_color=COLOR_THEME["card_bg"],
            segmented_button_selected_color="#032338",
            segmented_button_selected_hover_color="#00E5FF",
            segmented_button_unselected_color=COLOR_THEME["card_bg"],
            segmented_button_unselected_hover_color="#121D30",
            text_color=COLOR_THEME["text_main"],
            corner_radius=12
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=14, pady=(6, 6))
        self.tab_live = self.tabview.add("LIVE STREAM ANALYTICS")
        self.tab_db = self.tabview.add("IDENTITY DATABASE")
        self.tab_health = self.tabview.add("SYSTEM HEALTH")
        self.tab_settings = self.tabview.add("PARAMETERS")
        self._build_tab_live()
        self._build_tab_db()
        self._build_tab_health()
        self._build_tab_settings()
    def _build_tab_live(self) -> None:
        main = self.tab_live
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=3, uniform="cols")  
        main.grid_columnconfigure(1, weight=1, uniform="cols")  
        self.video_box = ctk.CTkFrame(
            main, fg_color="#010206", corner_radius=14,
            border_width=1, border_color="#121B2B"
        )
        self.video_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_label = tk.Label(
            self.video_box, bg="#010206", fg=COLOR_THEME["text_sub"], bd=0, highlightthickness=0, cursor="crosshair",
            text="[ NO LIVE SIGNAL ]\n\nClick 'START WEBCAM' or 'UPLOAD VIDEO'\nto launch real-time AI recognition engine",
            font=(MONO, 13, "bold"), justify="center",
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_label.bind("<Button-1>", self._on_video_click)

        panel = ctk.CTkScrollableFrame(
            main, fg_color=COLOR_THEME["card_bg"], corner_radius=14,
            border_width=1, border_color=COLOR_THEME["card_border"],
            scrollbar_button_color="#121D30", scrollbar_button_hover_color="#00E5FF"
        )
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        row = 0
        head_frame = ctk.CTkFrame(panel, fg_color="transparent")
        head_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=(10, 6)); row += 1
        ctk.CTkLabel(
            head_frame, text="TARGET ANALYTICS", font=(MONO, 12, "bold"), text_color=COLOR_THEME["accent_cyan"]
        ).pack(side="left")
        ctk.CTkLabel(
            head_frame, text="ACTIVE", font=(MONO, 9, "bold"), text_color=COLOR_THEME["accent_green"],
            fg_color="#064E3B", corner_radius=4, padx=6, pady=2
        ).pack(side="right")
        crop_frame = ctk.CTkFrame(
            panel, fg_color="#010206", width=CROP_SIZE + 4, height=CROP_SIZE + 4,
            corner_radius=12, border_width=1, border_color="#18253A"
        )
        crop_frame.grid(row=row, column=0, pady=(0, 8)); row += 1
        crop_frame.grid_propagate(False)
        crop_frame.grid_rowconfigure(0, weight=1)
        crop_frame.grid_columnconfigure(0, weight=1)
        self.crop_label = tk.Label(
            crop_frame, bg="#010206", fg=COLOR_THEME["text_sub"], bd=0, highlightthickness=0,
            text="NO TARGET", font=(MONO, 10, "bold")
        )
        self.crop_label.grid(row=0, column=0)
        ids = ctk.CTkFrame(panel, fg_color="#040711", corner_radius=8, border_width=1, border_color="#121D30")
        ids.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 4)); row += 1
        ids.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ids, text=" TARGET ID:", font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"]).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.id_label = ctk.CTkLabel(ids, text="—", font=(MONO, 14, "bold"), text_color=COLOR_THEME["text_main"])
        self.id_label.grid(row=0, column=1, sticky="e", padx=6, pady=4)
        ctk.CTkLabel(ids, text=" NAME/ALIAS:", font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"]).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.name_label = ctk.CTkLabel(
            ids, text="—", font=(MONO, 12, "bold"), text_color=COLOR_THEME["text_main"],
            wraplength=160, justify="right"
        )
        self.name_label.grid(row=1, column=1, sticky="e", padx=6, pady=4)
        self.state_badge = ctk.CTkLabel(
            panel, text="NO FACE DETECTED", font=(MONO, 11, "bold"), corner_radius=8,
            fg_color="#0F172A", text_color=COLOR_THEME["text_sub"], height=30
        )
        self.state_badge.grid(row=row, column=0, sticky="ew", padx=10, pady=(4, 4)); row += 1
        conf = ctk.CTkFrame(panel, fg_color="transparent")
        conf.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        conf.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            conf, text="MATCH CONFIDENCE", font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"]
        ).grid(row=0, column=0, sticky="w")
        self.conf_label = ctk.CTkLabel(conf, text="—", font=(MONO, 12, "bold"), text_color=COLOR_THEME["text_main"])
        self.conf_label.grid(row=0, column=1, sticky="e")
        self.conf_bar = ctk.CTkProgressBar(conf, height=6, progress_color=COLOR_THEME["accent_cyan"], fg_color="#0F172A")
        self.conf_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        self.conf_bar.set(0)

        pose = ctk.CTkFrame(panel, fg_color="transparent")
        pose.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        pose.grid_columnconfigure(1, weight=1)
        self.pose_widgets = {}
        for i, (key, title, limit) in enumerate((("yaw", "YAW", config.POSE_MAX_YAW),
                                                 ("pitch", "PITCH", config.POSE_MAX_PITCH),
                                                 ("roll", "ROLL", config.POSE_MAX_ROLL))):
            ctk.CTkLabel(
                pose, text=title, font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"], width=46, anchor="w"
            ).grid(row=i, column=0, sticky="w")
            bar = ctk.CTkProgressBar(pose, height=5, progress_color=COLOR_THEME["accent_green"], fg_color="#0F172A")
            bar.grid(row=i, column=1, sticky="ew", padx=6)
            bar.set(0)
            val = ctk.CTkLabel(pose, text="—", font=(MONO, 11, "bold"), text_color=COLOR_THEME["text_main"], width=58, anchor="e")
            val.grid(row=i, column=2, sticky="e")
            self.pose_widgets[key] = (val, bar, limit)
        self.quality_badge = ctk.CTkLabel(
            panel, text="—", font=(MONO, 11, "bold"), corner_radius=6,
            fg_color="#0F172A", text_color=COLOR_THEME["text_sub"], height=26,
            wraplength=230
        )
        self.quality_badge.grid(row=row, column=0, sticky="ew", padx=10, pady=(6, 2)); row += 1
        self.sharp_label = ctk.CTkLabel(panel, text="Sharpness: —", font=(MONO, 10), text_color=COLOR_THEME["text_sub"], anchor="w")
        self.sharp_label.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        self.votes_label = ctk.CTkLabel(
            panel, text="VERIFICATION TRACK", font=(MONO, 11, "bold"),
            text_color=COLOR_THEME["text_sub"], anchor="w"
        )
        self.votes_label.grid(row=row, column=0, sticky="ew", padx=10, pady=(4, 0)); row += 1
        self.feat_label = ctk.CTkLabel(
            panel, text="Features stored in DB: —", font=(MONO, 10),
            text_color=COLOR_THEME["text_main"], anchor="w", wraplength=240, justify="left"
        )
        self.feat_label.grid(row=row, column=0, sticky="ew", padx=10, pady=(2, 0)); row += 1
        self.db_label = ctk.CTkLabel(
            panel, text="Registered Persons in DB: 0", font=(MONO, 10, "bold"), text_color=COLOR_THEME["accent_purple"], anchor="w"
        )
        self.db_label.grid(row=row, column=0, sticky="ew", padx=10); row += 1

        enroll = ctk.CTkFrame(panel, fg_color="transparent")
        enroll.grid(row=row, column=0, sticky="ew", padx=10, pady=(6, 4)); row += 1
        enroll.grid_columnconfigure(0, weight=1)
        self.enroll_label = ctk.CTkLabel(
            enroll, text="System Status: Active Monitoring", font=(MONO, 10, "bold"),
            text_color=COLOR_THEME["text_sub"], anchor="w", wraplength=240, justify="left"
        )
        self.enroll_label.grid(row=0, column=0, sticky="ew")
        self.enroll_bar = ctk.CTkProgressBar(enroll, height=5, progress_color=COLOR_THEME["accent_green"], fg_color="#0F172A")
        self.enroll_bar.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.enroll_bar.set(0)
        log_head = ctk.CTkFrame(panel, fg_color="transparent")
        log_head.grid(row=row, column=0, sticky="ew", padx=10, pady=(6, 2)); row += 1
        log_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(log_head, text="SYSTEM AUDIT LOG", font=(MONO, 10, "bold"), text_color=COLOR_THEME["text_sub"]).grid(row=0, column=0, sticky="w")
        
        btn_export = ctk.CTkButton(
            log_head, text="EXPORT LOGS", font=(MONO, 9, "bold"), height=22, width=80,
            fg_color="#0E1A2E", hover_color="#00E5FF", text_color=COLOR_THEME["accent_cyan"],
            command=self._on_export_logs
        )
        btn_export.grid(row=0, column=1, sticky="e")
        self.log_box = ctk.CTkTextbox(
            panel, fg_color="#010206", text_color=COLOR_THEME["text_main"], font=(MONO, 9),
            border_width=1, border_color="#18253A", corner_radius=10,
            wrap="word", state="disabled", height=120
        )
        self.log_box.grid(row=row, column=0, sticky="ew", padx=10, pady=(0, 10)); row += 1
        for tag, colour in (("ok", COLOR_THEME["accent_green"]), ("warn", COLOR_THEME["accent_amber"]), ("error", COLOR_THEME["accent_red"]),
                            ("info", COLOR_THEME["accent_cyan"])):
            self.log_box.tag_config(tag, foreground=colour)
    def _build_tab_db(self) -> None:
        main = self.tab_db
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)
        top_bar = ctk.CTkFrame(main, fg_color=COLOR_THEME["card_bg"], corner_radius=10, border_width=1, border_color=COLOR_THEME["card_border"])
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top_bar, text="SEARCH DATABASE:", font=(MONO, 11, "bold"), text_color=COLOR_THEME["text_sub"]
                     ).grid(row=0, column=0, padx=12, pady=10)
        
        self.db_search_entry = ctk.CTkEntry(
            top_bar, placeholder_text="Filter by Person ID or Name...", font=(MONO, 11),
            fg_color="#040711", border_color="#18253A", height=34
        )
        self.db_search_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.db_search_entry.bind("<KeyRelease>", lambda e: self._refresh_db_table())
        btn_refresh = ctk.CTkButton(
            top_bar, text="REFRESH LIST", font=(MONO, 11, "bold"), height=34,
            fg_color="#032338", hover_color="#00E5FF", border_width=1, border_color=COLOR_THEME["accent_cyan"],
            command=self._refresh_db_table
        )
        btn_refresh.grid(row=0, column=2, padx=12)

        self.db_list_frame = ctk.CTkScrollableFrame(
            main, fg_color=COLOR_THEME["card_bg"], corner_radius=10,
            border_width=1, border_color=COLOR_THEME["card_border"]
        )
        self.db_list_frame.grid(row=1, column=0, sticky="nsew")
        self.db_list_frame.grid_columnconfigure(1, weight=1)

    def _refresh_db_table(self) -> None:
        for child in self.db_list_frame.winfo_children():
            child.destroy()
        filter_text = self.db_search_entry.get().strip().lower() if hasattr(self, "db_search_entry") else ""
        headers = ["ID", "REGISTERED NAME", "FEATURES COUNT", "STATUS", "ACTIONS"]
        for col, h in enumerate(headers):
            lbl = ctk.CTkLabel(self.db_list_frame, text=h, font=(MONO, 11, "bold"), text_color=COLOR_THEME["accent_cyan"])
            lbl.grid(row=0, column=col, sticky="w" if col < 4 else "e", padx=16, pady=10)

        ctk.CTkFrame(self.db_list_frame, fg_color=COLOR_THEME["card_border"], height=1).grid(row=1, column=0, columnspan=5, sticky="ew", padx=10)
        row_idx = 2
        num_items = 0
        try:
            persons = self.pipeline.db.list_persons()
            for p in persons:
                if isinstance(p, (tuple, list)):
                    person_id = p[0] if len(p) > 0 else 0
                    pname = p[1] if len(p) > 1 else "Unknown"
                    feat_count = p[2] if len(p) > 2 else 0
                elif isinstance(p, dict):
                    person_id = p.get('id', 0)
                    pname = p.get('name', 'Unknown')
                    feat_count = p.get('feature_count', 0)
                else:
                    continue

                pid_str = f"{int(person_id):05d}"
                if filter_text and (filter_text not in pid_str.lower() and filter_text not in pname.lower()):
                    continue
                num_items += 1
                ctk.CTkLabel(self.db_list_frame, text=pid_str, font=(MONO, 11, "bold"), text_color=COLOR_THEME["text_main"]
                             ).grid(row=row_idx, column=0, sticky="w", padx=16, pady=8)
                
                ctk.CTkLabel(self.db_list_frame, text=pname, font=(MONO, 11), text_color=COLOR_THEME["text_main"]
                             ).grid(row=row_idx, column=1, sticky="w", padx=16, pady=8)
                
                ctk.CTkLabel(self.db_list_frame, text=str(feat_count), font=(MONO, 11), text_color=COLOR_THEME["accent_purple"]
                             ).grid(row=row_idx, column=2, sticky="w", padx=16, pady=8)
                
                ctk.CTkLabel(self.db_list_frame, text="ENROLLED", font=(MONO, 10, "bold"), text_color=COLOR_THEME["accent_green"],
                             fg_color="#064E3B", corner_radius=4, padx=6, pady=2
                             ).grid(row=row_idx, column=3, sticky="w", padx=16, pady=8)

                btn_delete = ctk.CTkButton(
                    self.db_list_frame, text="DELETE", font=(MONO, 10, "bold"), height=26, width=70,
                    fg_color="#3B0808", hover_color="#EF4444", text_color="#FFFFFF",
                    command=lambda pid=person_id: self._on_delete_db_entry(pid)
                )
                btn_delete.grid(row=row_idx, column=4, sticky="e", padx=16, pady=8)
                row_idx += 1

        except Exception as exc:
            logger.exception("Error loading DB table")
        if num_items == 0:
            ctk.CTkLabel(self.db_list_frame, text="No registered identity records found.", font=(MONO, 11), text_color=COLOR_THEME["text_sub"]
                         ).grid(row=2, column=0, columnspan=5, pady=30)
    def _on_delete_db_entry(self, person_id: int) -> None:
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove Person ID {person_id:05d} from Database?"):
            try:
                self.pipeline.db.remove_person(person_id)
                self._log(f"Removed Person ID {person_id:05d} from Database.", "warn")
                self._refresh_db_table()
                self.db_label.configure(text=f"Registered Persons in DB: {self.pipeline.db.num_persons()}")
            except Exception as exc:
                messagebox.showerror("Error", f"Could not remove record: {exc}")

    def _build_tab_health(self) -> None:
        main = self.tab_health
        main.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="stat")
        main.grid_rowconfigure(1, weight=1)

        self.card_fps = StatCard(main, "SYSTEM FPS", "0.0", "FPS", COLOR_THEME["accent_purple"])
        self.card_fps.grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 10))

        self.card_latency = StatCard(main, "FRAME LATENCY", "0.0", "ms", COLOR_THEME["accent_cyan"])
        self.card_latency.grid(row=0, column=1, sticky="ew", padx=6, pady=(0, 10))

        self.card_db = StatCard(main, "IDENTITY RECORDS", "0", "Persons", COLOR_THEME["accent_green"])
        self.card_db.grid(row=0, column=2, sticky="ew", padx=6, pady=(0, 10))

        self.card_status = StatCard(main, "ENGINE STATE", "IDLE", "", COLOR_THEME["accent_amber"])
        self.card_status.grid(row=0, column=3, sticky="ew", padx=6, pady=(0, 10))

        health_box = ctk.CTkFrame(main, fg_color=COLOR_THEME["card_bg"], corner_radius=10, border_width=1, border_color=COLOR_THEME["card_border"])
        health_box.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=6)
        health_box.grid_columnconfigure(0, weight=1)
        health_box.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(health_box, text="SYSTEM DIAGNOSTICS & TELEMETRY", font=(MONO, 12, "bold"), text_color=COLOR_THEME["accent_cyan"]
                     ).grid(row=0, column=0, sticky="w", padx=16, pady=12)

        self.health_console = ctk.CTkTextbox(
            health_box, fg_color="#010206", text_color=COLOR_THEME["text_main"], font=(MONO, 10),
            border_width=1, border_color="#18253A", corner_radius=8, state="disabled"
        )
        self.health_console.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
    def _build_tab_settings(self) -> None:
        main = self.tab_settings
        main.grid_columnconfigure(0, weight=1)

        box = ctk.CTkFrame(main, fg_color=COLOR_THEME["card_bg"], corner_radius=10, border_width=1, border_color=COLOR_THEME["card_border"])
        box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        box.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(box, text="AI RECOGNITION PARAMETERS", font=(MONO, 13, "bold"), text_color=COLOR_THEME["accent_cyan"]
                     ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 12))

        ctk.CTkLabel(box, text="Cosine Similarity Threshold:", font=(MONO, 11), text_color=COLOR_THEME["text_sub"]
                     ).grid(row=1, column=0, sticky="w", padx=16, pady=10)
        
        self.slider_conf = ctk.CTkSlider(box, from_=0.3, to=0.8, number_of_steps=50, progress_color=COLOR_THEME["accent_cyan"])
        self.slider_conf.set(getattr(config, "SIMILARITY_THRESHOLD", 0.5))
        self.slider_conf.grid(row=1, column=1, sticky="ew", padx=16, pady=10)

        ctk.CTkLabel(box, text="Laplacian Blur Threshold:", font=(MONO, 11), text_color=COLOR_THEME["text_sub"]
                     ).grid(row=2, column=0, sticky="w", padx=16, pady=10)
        self.slider_blur = ctk.CTkSlider(box, from_=10.0, to=200.0, number_of_steps=38, progress_color=COLOR_THEME["accent_green"])
        self.slider_blur.set(getattr(config, "BLUR_THRESHOLD", 50.0))
        self.slider_blur.grid(row=2, column=1, sticky="ew", padx=16, pady=10)
        btn_save = ctk.CTkButton(
            box, text="APPLY PARAMETERS", font=(MONO, 11, "bold"), height=36,
            fg_color="#032338", hover_color="#00E5FF", border_width=1, border_color=COLOR_THEME["accent_cyan"],
            command=self._on_save_settings
        )
        btn_save.grid(row=3, column=0, columnspan=2, sticky="e", padx=16, pady=20)

    def _on_save_settings(self) -> None:
        new_conf = self.slider_conf.get()
        new_blur = self.slider_blur.get()
        config.SIMILARITY_THRESHOLD = new_conf
        config.BLUR_THRESHOLD = new_blur
        self._log(f"Updated Parameters: Similarity Threshold={new_conf:.2f}, Blur Threshold={new_blur:.0f}", "ok")
        messagebox.showinfo("Parameters Updated", "AI Engine parameters updated successfully.")
    def _build_controls(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=COLOR_THEME["card_bg"], corner_radius=14, border_width=1, border_color=COLOR_THEME["card_border"])
        bar.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        for col in range(4):
            bar.grid_columnconfigure(col, weight=1, uniform="btn")

        def make(text, bg_color, hover_color, border_color, text_color, command, col):
            btn = ctk.CTkButton(
                bar, text=text, command=command, height=48, font=(MONO, 12, "bold"),
                fg_color=bg_color, hover_color=hover_color, border_width=1,
                border_color=border_color, text_color=text_color, text_color_disabled="#475569",
                corner_radius=10
            )
            btn.grid(row=0, column=col, padx=10, pady=12, sticky="ew")
            return btn

        self.btn_upload = make("UPLOAD VIDEO", "#064E3B", "#047857", "#34D399", "#FFFFFF", self._on_upload, 0)
        self.btn_webcam = make("START WEBCAM", "#1E1B4B", "#3730A3", "#818CF8", "#FFFFFF", self._on_webcam, 1)
        self.btn_pause  = make("PAUSE STREAM",  "#451A03", "#B45309", "#FBBF24", "#FFFFFF", self._on_pause, 2)
        self.btn_stop   = make("STOP STREAM",   "#4C0519", "#BE123C", "#FB7185", "#FFFFFF", self._on_stop, 3)
            
    def _set_buttons_state(self) -> None:
        state = self.pipeline.state
        ready = self._models_ready and not self._starting
        active = state in (ST_RUNNING, ST_PAUSED)
        self.btn_upload.configure(state="normal" if ready else "disabled")
        self.btn_webcam.configure(state="normal" if ready else "disabled")
        self.btn_pause.configure(state="normal" if (ready and active) else "disabled",
                                 text="RESUME STREAM" if state == ST_PAUSED else "PAUSE STREAM")
        self.btn_stop.configure(state="normal" if (self._models_ready and (active or state in (ST_FINISHED, ST_ERROR)))
                                else "disabled")

    def _set_status(self, text: str, colour: str) -> None:
        if (text, colour) != self._status_cache:
            self._status_cache = (text, colour)
            self.status_label.configure(text=text, text_color=colour)
            if hasattr(self, "card_status"):
                self.card_status.update_val(text.replace("● ", "").replace("⏸ ", "").replace("■ ", ""))

    def _refresh_status(self) -> None:
        state, src = self.pipeline.state, self.pipeline.source_name
        if not self._models_ready:
            return
        if self._starting:
            self._set_status("OPENING VIDEO STREAM...", COLOR_THEME["accent_amber"])
        elif state == ST_RUNNING:
            self._set_status(f"LIVE | {src}", COLOR_THEME["accent_green"])
        elif state == ST_PAUSED:
            self._set_status("STREAM PAUSED", COLOR_THEME["accent_amber"])
        elif state == ST_FINISHED:
            self._set_status("VIDEO ENDED", COLOR_THEME["accent_amber"])
        elif state == ST_ERROR:
            self._set_status("STREAM ERROR", COLOR_THEME["accent_red"])
        else:
            self._set_status("SYSTEM READY", COLOR_THEME["accent_cyan"])
    def _log(self, text: str, level: str = "info") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._logs_cache.append((stamp, level, text))
        
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {text}\n", level)
        if int(self.log_box.index("end-1c").split(".")[0]) > 300:
            self.log_box.delete("1.0", "60.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

        if hasattr(self, "health_console"):
            self.health_console.configure(state="normal")
            self.health_console.insert("end", f"[{stamp}] [{level.upper()}] {text}\n")
            self.health_console.see("end")
            self.health_console.configure(state="disabled")

    def _on_export_logs(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export System Audit Logs",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("JSON File", "*.json")]
        )
        if path:
            try:
                if path.endswith(".json"):
                    data = [{"timestamp": t, "level": l, "message": m} for t, l, m in self._logs_cache]
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        for t, l, m in self._logs_cache:
                            f.write(f"[{t}] [{l.upper()}] {m}\n")
                messagebox.showinfo("Export Successful", f"Audit logs exported to:\n{path}")
            except Exception as exc:
                messagebox.showerror("Export Failed", f"Could not write log file: {exc}")

    def _load_models_bg(self) -> None:
        try:
            self.pipeline.load_models()
            self._ui_queue.put(("models_ready",))
        except ModelLoadError as exc:
            self._ui_queue.put(("models_error", str(exc)))
        except Exception as exc:  
            logger.exception("Model loading error")
            self._ui_queue.put(("models_error", f"Unexpected error while loading models: {exc}"))

    def _start_source(self, source, is_file: bool, label: str) -> None:
        if not self._models_ready:
            messagebox.showinfo("System Not Ready", "AI models are currently loading into memory, please wait...")
            return
        if self._starting:
            return
        self._starting = True
        self._set_buttons_state()
        self._refresh_status()

        def worker() -> None:
            try:
                self.pipeline.start(source, is_file=is_file, label=label)
                self._ui_queue.put(("started", label))
            except VideoSourceError as exc:
                self._ui_queue.put(("start_error", str(exc)))
            except Exception as exc: 
                logger.exception("Video source error")
                self._ui_queue.put(("start_error", f"Unexpected error: {exc}"))

        threading.Thread(target=worker, daemon=True, name="source-starter").start()

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                msg = self._ui_queue.get_nowait()
            except queue.Empty:
                return
            kind = msg[0]
            if kind == "models_ready":
                self._models_ready = True
                self._log(f"Models active | DB initialized with {self.pipeline.db.num_persons()} registered persons.", "ok")
                self.db_label.configure(text=f"Registered Persons in DB: {self.pipeline.db.num_persons()}")
                self.card_db.update_val(str(self.pipeline.db.num_persons()))
                self._refresh_db_table()
            elif kind == "models_error":
                self._set_status("MODEL INITIALIZATION FAILED", COLOR_THEME["accent_red"])
                self._log(msg[1], "error")
                messagebox.showerror("Model Load Error", msg[1])
            elif kind == "started":
                self._starting = False
                self._last_seq = -1
                self._log(f"Video source online: {msg[1]}", "ok")
            elif kind == "start_error":
                self._starting = False
                self._log(msg[1], "error")
                messagebox.showerror("Failed to Open Video Stream", msg[1])
            self._set_buttons_state()
            self._refresh_status()

    def _drain_events(self) -> None:
        for _ in range(50):
            try:
                level, text = self.pipeline.events.get_nowait()
            except queue.Empty:
                return
            if level == "error":
                self._last_error_text = text
            self._log(text, level)

    def _on_upload(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video Stream File",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"), ("All Files", "*.*")],
        )
        if path:
            self._reset_view()
            self._start_source(path, True, Path(path).name)

    def _on_webcam(self) -> None:
        self._reset_view()
        self._start_source(config.CAMERA_ID, False, f"Webcam #{config.CAMERA_ID}")

    def _on_pause(self) -> None:
        paused = self.pipeline.toggle_pause()
        self._log("Stream execution paused." if paused else "Resuming stream execution.", "info")
        self._set_buttons_state()
        self._refresh_status()

    def _on_stop(self) -> None:
        self.pipeline.stop()
        self._reset_view()
        self._log("Pipeline offline. Feature database saved to disk.", "info")
        self._set_buttons_state()
        self._refresh_status()

    def _on_add_person(self) -> None:
        pass

    def _on_video_click(self, event) -> None:
        result = self.pipeline.get_latest()
        if result is None:
            return
        track_id = overlay.pick_face(result, event.x, event.y, self._sx, self._sy)
        if track_id is not None:
            self.pipeline.select_track(track_id)
            self._last_seq = -1                    

    def _on_close(self) -> None:
        self._closing = True
        try:
            self.pipeline.shutdown()
        except Exception: 
            logger.exception("Error during application shutdown")
        self.destroy()

    def _tick(self) -> None:
        if self._closing:
            return
        start_time = time.time()
        try:
            self._drain_ui_queue()
            self._drain_events()

            state = self.pipeline.state
            if state != self._last_state:
                if state == ST_ERROR and self._last_error_text:
                    messagebox.showerror("Video Stream Error", self._last_error_text)
                self._last_state = state
                self._last_seq = -1                 
                self._set_buttons_state()
            self._refresh_status()

            result = self.pipeline.get_latest()
            if result is not None and result.seq != self._last_seq:
                self._last_seq = result.seq
                self._render_video(result)
                self._update_panel(result)
                self._update_fps()

            self._proc_time_ms = (time.time() - start_time) * 1000.0
            self.latency_label.configure(text=f"LATENCY: {self._proc_time_ms:.1f} ms")
            if hasattr(self, "card_latency"):
                self.card_latency.update_val(f"{self._proc_time_ms:.1f}")
        except Exception: 
            logger.exception("Error in GUI update loop execution")
        finally:
            if not self._closing:
                self.after(config.UI_REFRESH_MS, self._tick)

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = now
            self.fps_label.configure(text=f"FPS: {self._current_fps:.1f}")
            if hasattr(self, "card_fps"):
                self.card_fps.update_val(f"{self._current_fps:.1f}")

    def _render_video(self, result: FrameResult) -> None:
        state = self.pipeline.state
        banner = {"paused": "PAUSED", "finished": "STREAM FINISHED"}.get(state, "")
        img, self._sx, self._sy = overlay.render_frame(
            result, self.video_box.winfo_width() - 4, self.video_box.winfo_height() - 4,
            state_text=state.upper(), source_name=self.pipeline.source_name, banner=banner,
        )
        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo               

    def _reset_view(self) -> None:
        self._last_seq = -1
        self.fps_label.configure(text="FPS: --")
        self.latency_label.configure(text="LATENCY: -- ms")
        self.video_label.configure(
            image="",
            text="[ NO LIVE SIGNAL ]\n\nClick 'START WEBCAM' or 'UPLOAD VIDEO'\nto launch real-time AI recognition engine"
        )
        self.video_label.image = None
        self._show_crop(None)
        self._clear_panel()
    def _show_crop(self, crop) -> None:
        if crop is None:
            self.crop_label.configure(image="", text="NO TARGET")
            self.crop_label.image = None
            return
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img.thumbnail((CROP_SIZE, CROP_SIZE))
        photo = ImageTk.PhotoImage(img)
        self.crop_label.configure(image=photo, text="")
        self.crop_label.image = photo

    def _clear_panel(self) -> None:
        self.id_label.configure(text="—", text_color=COLOR_THEME["text_main"])
        self.name_label.configure(text="—", text_color=COLOR_THEME["text_main"])
        self.state_badge.configure(text="NO FACE DETECTED", fg_color="#0F172A", text_color=COLOR_THEME["text_sub"])
        self.conf_label.configure(text="—")
        self.conf_bar.set(0)
        for val, bar, _ in self.pose_widgets.values():
            val.configure(text="—")
            bar.set(0)
        self.quality_badge.configure(text="—", fg_color="#0F172A", text_color=COLOR_THEME["text_sub"])
        self.sharp_label.configure(text="Sharpness: —")
        self.votes_label.configure(text="VERIFICATION TRACK", text_color=COLOR_THEME["text_sub"])
        self.feat_label.configure(text="Features stored in DB: —")
        self.enroll_label.configure(text="System Status: Active Monitoring", text_color=COLOR_THEME["text_sub"])
        self.enroll_bar.set(0)

    def _update_panel(self, result: FrameResult) -> None:
        self.db_label.configure(text=f"Registered Persons in DB: {result.db_size}")
        if hasattr(self, "card_db"):
            self.card_db.update_val(str(result.db_size))

        enroll = result.enroll
        if enroll.active:
            self.enroll_label.configure(
                text=f"MONITORING '{enroll.name}': {enroll.progress}/{enroll.total} samples",
                text_color=COLOR_THEME["accent_green"]
            )
            self.enroll_bar.set(enroll.progress / max(enroll.total, 1))
        else:
            msg = enroll.message
            if msg:
                if "Đã thêm" in msg:
                    msg = msg.replace("Đã thêm", "Added").replace("vào CSDL", "to DB")
            self.enroll_label.configure(
                text=msg or "System Status: Active Monitoring",
                text_color=COLOR_THEME["accent_green"] if msg else COLOR_THEME["text_sub"]
            )
            self.enroll_bar.set(1.0 if msg and ("Added" in msg or "Đã thêm" in msg) else 0)

        face = next((f for f in result.faces if f.track_id == result.selected_track_id), None)
        self._show_crop(face.crop if face is not None else None)
        if face is None:
            self._clear_panel_face_only()
            return

        known = face.state == STATE_KNOWN
        unknown = face.state == STATE_UNKNOWN
        colour = overlay.state_color(face.state)
        self.id_label.configure(text=f"{face.person_id:05d}" if known else "—",
                                text_color=colour if known else COLOR_THEME["text_main"])
        auto_named = known and face.name.startswith(config.AUTO_NAME_PREFIX)
        self.name_label.configure(
            text=("(unnamed)" if auto_named else face.name) if known else ("Unknown" if unknown else "…"),
            text_color=colour)
        if known:
            badge = "VERIFIED TARGET"
        elif unknown:
            badge = (f"UNKNOWN | learning {face.learn[0]}/{face.learn[1]}" if face.learn
                     else "UNKNOWN TARGET")
        else:
            badge = "VERIFYING TARGET"
        self.state_badge.configure(text=badge, fg_color="#0F172A", text_color=colour)

        if known:
            self.conf_label.configure(text=f"{face.confidence * 100:.1f}%")
            self.conf_bar.set(max(0.0, min(1.0, face.confidence)))
        else:
            self.conf_label.configure(text="—")
            self.conf_bar.set(0)

        for key, value in (("yaw", face.yaw), ("pitch", face.pitch), ("roll", face.roll)):
            val, bar, limit = self.pose_widgets[key]
            if value is None:
                val.configure(text="—")
                bar.set(0)
                bar.configure(progress_color=COLOR_THEME["text_sub"])
            else:
                val.configure(text=f"{value:+.1f}°")
                bar.set(min(abs(value) / limit, 1.0))
                bar.configure(progress_color=COLOR_THEME["accent_green"] if abs(value) <= limit else COLOR_THEME["accent_red"])

        if face.is_good:
            self.quality_badge.configure(text="VALID QUALITY FRAME", text_color=COLOR_THEME["accent_green"])
        else:
            self.quality_badge.configure(text=f"FILTERED | {face.reason or 'unknown'}", text_color=COLOR_THEME["accent_red"])
        self.sharp_label.configure(text=f"Sharpness (Laplacian): {face.sharpness:.0f}  (>= {config.BLUR_THRESHOLD:.0f})")

        run = face.vote_run if not (known or unknown) else face.n_confirm
        self.votes_label.configure(
            text=f"VERIFICATION TRACK ({run}/{face.n_confirm})",
            text_color=colour
        )
        self.feat_label.configure(
            text=(f"Features stored in DB: {face.feature_count}" if known
                  else "Features stored in DB: — (not in DB)")
        )

    def _clear_panel_face_only(self) -> None:
        self.id_label.configure(text="—", text_color=COLOR_THEME["text_main"])
        self.name_label.configure(text="—", text_color=COLOR_THEME["text_main"])
        self.state_badge.configure(text="NO FACE DETECTED", fg_color="#0F172A", text_color=COLOR_THEME["text_sub"])
        self.conf_label.configure(text="—")
        self.conf_bar.set(0)
        for val, bar, _ in self.pose_widgets.values():
            val.configure(text="—")
            bar.set(0)
        self.quality_badge.configure(text="—", text_color=COLOR_THEME["text_sub"])
        self.sharp_label.configure(text="Sharpness: —")
        self.votes_label.configure(text="VERIFICATION TRACK", text_color=COLOR_THEME["text_sub"])
        self.feat_label.configure(text="Features stored in DB: —")