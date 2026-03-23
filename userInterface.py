# -*- coding: utf-8 -*-
"""
Galaxy Classifier AI — Tkinter GUI
3-page app: Upload → Processing → Results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import random
import asyncio
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import classification

# Load environment variables from script directory
script_dir = Path(__file__).parent
env_file = script_dir / '.env'
load_dotenv(env_file)

# ─────────────────────────────────────────────
#  Colour palette & font constants
# ─────────────────────────────────────────────
BG_DARK      = "#0D0F1A"   # deep space
BG_MID       = "#141828"   # card background
BG_CARD      = "#1C2235"   # elevated card
ACCENT       = "#7C6BFF"   # violet/purple
ACCENT2      = "#00D4AA"   # teal/mint
ACCENT_DIM   = "#3D3680"
TEXT_PRIMARY = "#E8E6F0"
TEXT_SEC     = "#8A87A0"
TEXT_HINT    = "#4A476A"
SUCCESS      = "#00D4AA"
WARNING      = "#F5A623"
DANGER       = "#FF5C6A"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEADER = ("Segoe UI", 14, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)

GALAXY_LABELS = {
    0: ("Elliptical",  "⬭",  "#F5A623"),
    1: ("Spiral",      "🌀",  "#7C6BFF"),
    2: ("Bar-Spiral",  "—",  "#00D4AA"),
}

WIN_W, WIN_H = 860, 660


# ─────────────────────────────────────────────
#  Helper: rounded rectangle on Canvas
# ─────────────────────────────────────────────
def rounded_rect(canvas, x1, y1, x2, y2, r=12, **kwargs):
    pts = [
        x1+r, y1,   x2-r, y1,
        x2, y1,     x2, y1+r,
        x2, y2-r,   x2, y2,
        x2-r, y2,   x1+r, y2,
        x1, y2,     x1, y2-r,
        x1, y1+r,   x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


# ─────────────────────────────────────────────
#  Validation & Standardization Helpers
# ─────────────────────────────────────────────
def standardize_image(image_input, output_size=(256, 256)):
    """Standardize image to 256x256 RGB format."""
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input.copy()
    
    # Convert to RGB (handles RGBA, CMYK, grayscale, etc.)
    if image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to output_size
    image_resized = image.resize(output_size, Image.Resampling.LANCZOS)
    
    # Ensure 8-bit RGB
    if image_resized.mode != 'RGB':
        image_resized = image_resized.convert('RGB')
    
    return image_resized


def load_image_b64(path: Path) -> tuple:
    """Load image and return (mime_type, base64_data)."""
    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp"
    }
    mime = mime_map.get(ext, "image/jpeg")
    b64_data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return mime, b64_data


def validate_galaxy(image_path: str) -> dict:
    """
    Validate if image is a galaxy using OpenAI GPT-5.4.
    Returns dict with is_galaxy (bool), response (str), error (str or None)
    """
    try:
        # Import openai
        try:
            from openai import OpenAI
        except ImportError:
            return {
                "is_galaxy": None,
                "response": None,
                "error": "openai not installed. Install with: pip install openai",
            }
        
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        
        if not api_key:
            return {
                "is_galaxy": None,
                "response": None,
                "error": "No OPENAI_API_KEY found in .env file"
            }
        
        prompt = (
            "Examine the following image CAREFULLY. "
            "Determine if this is an image of a galaxy (e.g elliptical, spiral, irregular). "
            "Reply with EXACTLY one of the following options (DO NOT ADD EXTRA EXPLANATION): \n"
            "GALAXY - if the image shows a galaxy/galaxies. \n"
            "NOT GALAXY - if the image does NOT show a galaxy/galaxies. \n"
        )
        
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)
            
            # Read image and encode as base64
            path = Path(image_path)
            mime_type, b64_data = load_image_b64(path)
            
            # Call GPT-5.4
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_data}",
                                },
                            },
                        ],
                    }
                ],
            )
            
            text = response.choices[0].message.content.strip()
            is_galaxy = text.upper().startswith("GALAXY")
            
            return {
                "is_galaxy": is_galaxy,
                "response": text,
                "error": None,
            }
        
        except Exception as e:
            return {
                "is_galaxy": None,
                "response": None,
                "error": str(e),
            }
    
    except Exception as e:
        return {
            "is_galaxy": None,
            "response": None,
            "error": f"Exception: {str(e)}",
        }



# ─────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────
class GalaxyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Galaxy Classifier AI")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG_DARK)

        # Shared state
        self.image_path   = tk.StringVar()
        self.result_label = tk.StringVar()
        self.result_conf  = tk.DoubleVar()
        self.all_probs    = {}
        self.standardized_image_path = None
        self.standardized_pil_image = None

        # Container for stacked pages
        container = tk.Frame(self, bg=BG_DARK)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for PageClass in (UploadPage, ProcessingPage, ResultPage):
            page = PageClass(container, self)
            self.pages[PageClass.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("UploadPage")

    def show_page(self, name):
        page = self.pages[name]
        page.tkraise()
        if hasattr(page, "on_show"):
            page.on_show()


# ─────────────────────────────────────────────
#  Page 1 — Upload
# ─────────────────────────────────────────────
class UploadPage(tk.Frame):
    def __init__(self, parent, app: GalaxyApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._build()

    def _build(self):
        # ── Top bar ──────────────────────────
        topbar = tk.Frame(self, bg=BG_MID, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="✦  GALAXY CLASSIFIER",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_MID, fg=ACCENT).pack(side="left", padx=20)

        # Step indicator
        self._step_bar(topbar, current=0)

        # ── Body ─────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=40, pady=30)

        # Title
        tk.Label(body, text="Upload a Galaxy Image",
                 font=FONT_TITLE, bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(body, text="Select a .jpg or .png image to classify its morphology",
                 font=FONT_BODY, bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(4, 20))

        # Drop / preview zone (canvas)
        self.canvas = tk.Canvas(body, width=360, height=300,
                                bg=BG_CARD, highlightthickness=0)
        self.canvas.pack(side="left")
        self._draw_dropzone()
        self.canvas.bind("<Button-1>", lambda e: self._pick_file())

        # Right panel
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(24, 0))

        # File info card
        info_card = tk.Frame(right, bg=BG_CARD, padx=16, pady=16)
        info_card.pack(fill="x", pady=(0, 16))

        tk.Label(info_card, text="SELECTED FILE",
                 font=FONT_SMALL, bg=BG_CARD, fg=TEXT_HINT).pack(anchor="w")
        self.lbl_path = tk.Label(info_card, text="No file selected",
                                 font=FONT_MONO, bg=BG_CARD, fg=TEXT_SEC,
                                 wraplength=340, justify="left")
        self.lbl_path.pack(anchor="w", pady=(4, 0))

        # Instructions
        instr = tk.Frame(right, bg=BG_CARD, padx=16, pady=14)
        instr.pack(fill="x", pady=(0, 20))

        tk.Label(instr, text="SUPPORTED FORMATS",
                 font=FONT_SMALL, bg=BG_CARD, fg=TEXT_HINT).pack(anchor="w")
        for line in ["• JPEG / JPG", "• PNG", "• BMP / TIFF"]:
            tk.Label(instr, text=line, font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC
                     ).pack(anchor="w")

        # Info note
        tk.Label(instr, text="\nModel info",
                 font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w")
        tk.Label(instr,
                 text="3-class CNN trained on Galaxy Zoo 2\n"
                      "classes: Elliptical · Spiral · Bar-Spiral",
                 font=FONT_SMALL, bg=BG_CARD, fg=TEXT_HINT,
                 justify="left").pack(anchor="w")

        # Buttons
        btn_row = tk.Frame(right, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(0, 8))

        tk.Button(btn_row, text="  Browse file…",
                  font=FONT_BODY, bg=ACCENT, fg="white",
                  activebackground=ACCENT_DIM, activeforeground="white",
                  relief="flat", padx=20, pady=8, cursor="hand2",
                  command=self._pick_file).pack(side="left")

        tk.Button(btn_row, text="Clear",
                  font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC,
                  activebackground=BG_MID, activeforeground=TEXT_PRIMARY,
                  relief="flat", padx=16, pady=8, cursor="hand2",
                  command=self._clear).pack(side="left", padx=(10, 0))

        self.btn_next = tk.Button(right, text="Classify  →",
                                  font=("Segoe UI", 12, "bold"),
                                  bg=ACCENT2, fg=BG_DARK,
                                  activebackground="#00B090", activeforeground=BG_DARK,
                                  relief="flat", padx=28, pady=12, width=18, height=2,
                                  cursor="hand2",
                                  state="disabled",
                                  command=self._go_process)
        self.btn_next.pack(anchor="e", pady=(18, 12))

    # ── helpers ──────────────────────────────
    def _step_bar(self, parent, current):
        steps = ["Upload", "Processing", "Results"]
        f = tk.Frame(parent, bg=BG_MID)
        f.pack(side="right", padx=20)
        for i, s in enumerate(steps):
            col = ACCENT if i == current else (TEXT_SEC if i < current else TEXT_HINT)
            tk.Label(f, text=f"{'●' if i == current else '○'}  {s}",
                     font=FONT_SMALL, bg=BG_MID, fg=col).pack(side="left", padx=8)

    def _draw_dropzone(self):
        self.canvas.delete("all")
        w, h = 360, 300
        # dashed border
        for i in range(0, w, 12):
            self.canvas.create_line(i, 0, min(i+6, w), 0, fill=TEXT_HINT)
            self.canvas.create_line(i, h-1, min(i+6, w), h-1, fill=TEXT_HINT)
        for i in range(0, h, 12):
            self.canvas.create_line(0, i, 0, min(i+6, h), fill=TEXT_HINT)
            self.canvas.create_line(w-1, i, w-1, min(i+6, h), fill=TEXT_HINT)
        # icon + text
        self.canvas.create_text(w//2, h//2 - 28, text="🌌",
                                font=("Segoe UI", 42), fill=TEXT_HINT)
        self.canvas.create_text(w//2, h//2 + 28,
                                text="Click to browse\nor drop an image here",
                                font=FONT_BODY, fill=TEXT_HINT,
                                justify="center")

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select galaxy image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")]
        )
        if path:
            self._load_preview(path)

    def _load_preview(self, path):
        self.app.image_path.set(path)
        self.lbl_path.config(text=os.path.basename(path), fg=TEXT_PRIMARY)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 300))
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            cw, ch = 360, 300
            iw, ih = img.size
            self.canvas.create_image((cw-iw)//2, (ch-ih)//2,
                                     anchor="nw", image=self._tk_img)
            self.btn_next.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

    def _clear(self):
        self.app.image_path.set("")
        self.lbl_path.config(text="No file selected", fg=TEXT_SEC)
        self._draw_dropzone()
        self.btn_next.config(state="disabled")

    def _go_process(self):
        if not self.app.image_path.get():
            return
        self.app.show_page("ProcessingPage")


# ─────────────────────────────────────────────
#  Page 2 — Processing
# ─────────────────────────────────────────────
class ProcessingPage(tk.Frame):
    def __init__(self, parent, app: GalaxyApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._anim_id = None
        self._step_idx = 0
        self._build()

    def _build(self):
        # ── Top bar ──────────────────────────
        topbar = tk.Frame(self, bg=BG_MID, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="✦  GALAXY CLASSIFIER",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_MID, fg=ACCENT).pack(side="left", padx=20)
        self._step_bar(topbar, current=1)

        # ── Body ─────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        # Left — animated canvas
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="y", padx=40, pady=30)

        tk.Label(left, text="Analysing…", font=FONT_TITLE,
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(left, text="Running CNN inference pipeline",
                 font=FONT_BODY, bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(4, 16))

        self.anim_canvas = tk.Canvas(left, width=320, height=220,
                                     bg=BG_CARD, highlightthickness=0)
        self.anim_canvas.pack()

        # Progress bar
        prog_frame = tk.Frame(left, bg=BG_DARK)
        prog_frame.pack(fill="x", pady=(14, 0))
        style = ttk.Style()
        style.theme_use("default")
        style.configure("G.Horizontal.TProgressbar",
                        troughcolor=BG_CARD, background=ACCENT,
                        thickness=6, borderwidth=0)
        self.progress = ttk.Progressbar(prog_frame, style="G.Horizontal.TProgressbar",
                                        orient="horizontal", length=320, mode="determinate")
        self.progress.pack()
        self.lbl_pct = tk.Label(prog_frame, text="0 %",
                                font=FONT_SMALL, bg=BG_DARK, fg=TEXT_SEC)
        self.lbl_pct.pack(anchor="e", pady=(2, 0))

        # Right — step log
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(0, 40), pady=30)

        tk.Label(right, text="Pipeline steps",
                 font=FONT_HEADER, bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w", pady=(34, 10))

        self.step_labels = []
        steps = [
            "Validate image (OpenAI API)",
            "Standardize to 256 × 256",
            "Conv block 1 — 32 filters",
            "Conv block 2 — 64 filters",
            "Conv block 3 — 128 filters",
            "Global average pooling",
            "Dense layer + Dropout",
            "Softmax classification",
        ]
        for s in steps:
            row = tk.Frame(right, bg=BG_DARK)
            row.pack(anchor="w", pady=2)
            dot = tk.Label(row, text="○", font=FONT_BODY, bg=BG_DARK, fg=TEXT_HINT, width=2)
            dot.pack(side="left")
            lbl = tk.Label(row, text=s, font=FONT_BODY, bg=BG_DARK, fg=TEXT_HINT)
            lbl.pack(side="left")
            self.step_labels.append((dot, lbl))

        # Back button
        tk.Button(right, text="← Back",
                  font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC,
                  activebackground=BG_MID, activeforeground=TEXT_PRIMARY,
                  relief="flat", padx=14, pady=6, cursor="hand2",
                  command=self._go_back).pack(anchor="w", pady=(12, 0))

    def _step_bar(self, parent, current):
        steps = ["Upload", "Processing", "Results"]
        f = tk.Frame(parent, bg=BG_MID)
        f.pack(side="right", padx=20)
        for i, s in enumerate(steps):
            col = ACCENT if i == current else (TEXT_SEC if i < current else TEXT_HINT)
            tk.Label(f, text=f"{'●' if i == current else '○'}  {s}",
                     font=FONT_SMALL, bg=BG_MID, fg=col).pack(side="left", padx=8)

    def on_show(self):
        self._step_idx = 0
        self.progress["value"] = 0
        self.lbl_pct.config(text="0 %")
        for dot, lbl in self.step_labels:
            dot.config(text="○", fg=TEXT_HINT)
            lbl.config(fg=TEXT_HINT)
        self._draw_neural_anim()
        self._run_mock_inference()

    # ── Animated dot-grid (fake neural net) ──
    def _draw_neural_anim(self):
        c = self.anim_canvas
        c.delete("all")
        layers = [3, 5, 5, 4, 3]
        lx = [15, 85, 155, 225, 290]
        ly_start = 30
        spacing = 36

        nodes = []
        for li, (lx_, n) in enumerate(zip(lx, layers)):
            col_nodes = []
            for ni in range(n):
                y = ly_start + ni * spacing + (max(layers) - n) * spacing // 2
                col_nodes.append((lx_, y))
            nodes.append(col_nodes)

        self._anim_nodes = nodes
        self._anim_edges = []
        for li in range(len(nodes) - 1):
            for (x1, y1) in nodes[li]:
                for (x2, y2) in nodes[li+1]:
                    eid = c.create_line(x1+10, y1+10, x2+10, y2+10,
                                        fill=TEXT_HINT, width=0.5)
                    self._anim_edges.append(eid)

        self._anim_ovals = []
        for col in nodes:
            for (x, y) in col:
                oid = c.create_oval(x, y, x+20, y+20,
                                    fill=BG_MID, outline=TEXT_HINT, width=1)
                self._anim_ovals.append(oid)

        self._pulse_counter = 0
        self._pulse()

    def _pulse(self):
        c = self.anim_canvas
        all_nodes = [n for col in self._anim_nodes for n in col]
        i = self._pulse_counter % len(all_nodes)
        x, y = all_nodes[i]
        for j, oid in enumerate(self._anim_ovals):
            c.itemconfig(oid, fill=ACCENT if j == i else BG_MID,
                         outline=ACCENT if j == i else TEXT_HINT)
        self._pulse_counter += 1
        self._anim_id = self.after(80, self._pulse)

    # ── Mock inference with progress ─────────
    def _run_mock_inference(self):
        def run():
            n = len(self.step_labels)
            image_path = self.app.image_path.get()
            
            # Step 0: Validate image
            dot_val, lbl_val = self.step_labels[0]
            self.after(0, lambda d=dot_val, l=lbl_val: (
                d.config(text="◉", fg=ACCENT),
                l.config(fg=TEXT_PRIMARY)
            ))
            
            # Run validation
            validation_result = validate_galaxy(image_path)
            
            pct = int(0.5 / n * 100)
            self.after(0, lambda p=pct: self._set_progress(p))
            import time; time.sleep(0.3)
            
            # Check validation result
            if validation_result["error"]:
                self.after(0, lambda d=dot_val, l=lbl_val: (
                    d.config(text="✗", fg=DANGER),
                    l.config(fg=DANGER)
                ))
                error_msg = f"Validation Error:\n{validation_result['error']}"
                self.after(100, lambda: messagebox.showerror("Validation Failed", error_msg))
                self.after(200, lambda: self.app.show_page("UploadPage"))
                return
            
            if not validation_result["is_galaxy"]:
                self.after(0, lambda d=dot_val, l=lbl_val: (
                    d.config(text="✗", fg=WARNING),
                    l.config(fg=WARNING)
                ))
                error_msg = f"This image does not appear to be a galaxy.\n\nResponse: {validation_result['response']}"
                self.after(100, lambda: messagebox.showwarning("Not a Galaxy", error_msg))
                self.after(200, lambda: self.app.show_page("UploadPage"))
                return
            
            # Mark validation as done
            self.after(0, lambda d=dot_val, l=lbl_val: (
                d.config(text="●", fg=ACCENT2),
                l.config(fg=TEXT_SEC)
            ))
            
            # Step 1: Standardize image
            dot_std, lbl_std = self.step_labels[1]
            self.after(0, lambda d=dot_std, l=lbl_std: (
                d.config(text="◉", fg=ACCENT),
                l.config(fg=TEXT_PRIMARY)
            ))
            
            try:
                standardized_img = standardize_image(image_path, output_size=(256, 256))
                # Save standardized image temporarily
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, "galaxy_standardized.png")
                standardized_img.save(temp_path)
                self.app.standardized_image_path = temp_path
                self.app.standardized_pil_image = standardized_img
            except Exception as e:
                self.after(0, lambda d=dot_std, l=lbl_std: (
                    d.config(text="✗", fg=DANGER),
                    l.config(fg=DANGER)
                ))
                error_msg = f"Standardization Error:\n{str(e)}"
                self.after(100, lambda: messagebox.showerror("Standardization Failed", error_msg))
                self.after(200, lambda: self.app.show_page("UploadPage"))
                return
            
            pct = int(1.5 / n * 100)
            self.after(0, lambda p=pct: self._set_progress(p))
            import time; time.sleep(0.3)
            
            # Mark standardization as done
            self.after(0, lambda d=dot_std, l=lbl_std: (
                d.config(text="●", fg=ACCENT2),
                l.config(fg=TEXT_SEC)
            ))
            
            # Run actual CNN classification on the standardized image
            cnn_result = classification.classify(self.app.standardized_pil_image)

            if cnn_result["error"]:
                self.after(100, lambda: messagebox.showerror(
                    "Classification Failed", f"CNN Error:\n{cnn_result['error']}"))
                self.after(200, lambda: self.app.show_page("UploadPage"))
                return

            self._cnn_result = cnn_result

            # Animate CNN layer steps (cosmetic — model has already run)
            for i in range(2, n):
                dot, lbl = self.step_labels[i]
                self.after(0, lambda d=dot, l=lbl: (
                    d.config(text="◉", fg=ACCENT),
                    l.config(fg=TEXT_PRIMARY)
                ))
                pct = int((i + 0.5) / n * 100)
                self.after(0, lambda p=pct: self._set_progress(p))
                import time; time.sleep(0.35)
                self.after(0, lambda d=dot, l=lbl: (
                    d.config(text="●", fg=ACCENT2),
                    l.config(fg=TEXT_SEC)
                ))

            self.after(0, lambda: self._set_progress(100))
            self.after(200, self._finish_inference)

        threading.Thread(target=run, daemon=True).start()

    def _set_progress(self, pct):
        self.progress["value"] = pct
        self.lbl_pct.config(text=f"{pct} %")

    def _finish_inference(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)

        result = self._cnn_result
        self.app.all_probs = result["probabilities"]
        self.app.result_label.set(result["label"])
        self.app.result_conf.set(result["confidence"])

        self.after(400, lambda: self.app.show_page("ResultPage"))

    def _go_back(self):
        if self._anim_id:
            self.after_cancel(self._anim_id)
        self.app.show_page("UploadPage")


# ─────────────────────────────────────────────
#  Page 3 — Results
# ─────────────────────────────────────────────
class ResultPage(tk.Frame):
    def __init__(self, parent, app: GalaxyApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._build()

    def _build(self):
        # ── Top bar ──────────────────────────
        topbar = tk.Frame(self, bg=BG_MID, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="✦  GALAXY CLASSIFIER",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_MID, fg=ACCENT).pack(side="left", padx=20)
        self._step_bar(topbar, current=2)

        # ── Body ─────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=40, pady=24)

        tk.Label(body, text="Classification Result",
                 font=FONT_TITLE, bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(body, text="CNN morphology prediction complete",
                 font=FONT_BODY, bg=BG_DARK, fg=TEXT_SEC).pack(anchor="w", pady=(4, 18))

        # ── Columns ──────────────────────────
        cols = tk.Frame(body, bg=BG_DARK)
        cols.pack(fill="both", expand=True)

        # Left — image thumbnail + prediction badge
        left = tk.Frame(cols, bg=BG_DARK)
        left.pack(side="left", fill="y")

        self.thumb_canvas = tk.Canvas(left, width=260, height=220,
                                      bg=BG_CARD, highlightthickness=0)
        self.thumb_canvas.pack()

        badge = tk.Frame(left, bg=BG_CARD, padx=16, pady=10)
        badge.pack(fill="x", pady=(12, 0))
        self.lbl_icon = tk.Label(badge, text="", font=("Segoe UI", 30),
                                 bg=BG_CARD, fg=ACCENT)
        self.lbl_icon.pack()
        self.lbl_pred = tk.Label(badge, text="—",
                                 font=("Segoe UI", 16, "bold"),
                                 bg=BG_CARD, fg=TEXT_PRIMARY)
        self.lbl_pred.pack()
        self.lbl_conf = tk.Label(badge, text="",
                                 font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC)
        self.lbl_conf.pack()

        # Right — probability bars + info
        right = tk.Frame(cols, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(24, 0))

        tk.Label(right, text="Class probabilities",
                 font=FONT_HEADER, bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        # Bar chart (canvas)
        self.bar_canvas = tk.Canvas(right, width=440, height=130,
                                    bg=BG_DARK, highlightthickness=0)
        self.bar_canvas.pack(anchor="w")

        # Description card
        self.desc_card = tk.Frame(right, bg=BG_CARD, padx=16, pady=12)
        self.desc_card.pack(fill="x", pady=(14, 0))
        tk.Label(self.desc_card, text="GALAXY TYPE INFO",
                 font=FONT_SMALL, bg=BG_CARD, fg=TEXT_HINT).pack(anchor="w")
        self.lbl_desc = tk.Label(self.desc_card, text="",
                                 font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC,
                                 wraplength=380, justify="left")
        self.lbl_desc.pack(anchor="w", pady=(4, 0))

        # ── Button row ───────────────────────
        btn_row = tk.Frame(body, bg=BG_DARK)
        btn_row.pack(fill="x", pady=(16, 0))

        tk.Button(btn_row, text="← Try another image",
                  font=FONT_BODY, bg=BG_CARD, fg=TEXT_SEC,
                  activebackground=BG_MID, activeforeground=TEXT_PRIMARY,
                  relief="flat", padx=18, pady=20, cursor="hand2",
                  command=self._go_back).pack(side="left")

        tk.Button(btn_row, text="Save result",
                  font=FONT_BODY, bg=ACCENT, fg="white",
                  activebackground=ACCENT_DIM, activeforeground="white",
                  relief="flat", padx=18, pady=20, cursor="hand2",
                  command=self._save_result).pack(side="left", padx=(10, 0))

    def _step_bar(self, parent, current):
        steps = ["Upload", "Processing", "Results"]
        f = tk.Frame(parent, bg=BG_MID)
        f.pack(side="right", padx=20)
        for i, s in enumerate(steps):
            col = ACCENT if i == current else (TEXT_SEC if i < current else TEXT_HINT)
            tk.Label(f, text=f"{'●' if i == current else '○'}  {s}",
                     font=FONT_SMALL, bg=BG_MID, fg=col).pack(side="left", padx=8)

    def on_show(self):
        label = self.app.result_label.get()
        conf  = self.app.result_conf.get()
        probs = self.app.all_probs
        path  = self.app.image_path.get()

        # Thumbnail - use standardized image if available
        self.thumb_canvas.delete("all")
        try:
            # Prefer standardized image if available
            if hasattr(self.app, 'standardized_pil_image') and self.app.standardized_pil_image:
                img = self.app.standardized_pil_image.copy()
            else:
                img = Image.open(path).convert("RGB")
            
            img.thumbnail((260, 220))
            self._tk_thumb = ImageTk.PhotoImage(img)
            iw, ih = img.size
            self.thumb_canvas.create_image((260-iw)//2, (220-ih)//2,
                                           anchor="nw", image=self._tk_thumb)
        except Exception:
            self.thumb_canvas.create_text(130, 110, text="(preview unavailable)",
                                          fill=TEXT_HINT, font=FONT_BODY)

        # Badge
        icons = {"Elliptical": "⬭", "Spiral": "🌀", "Bar-Spiral": "⊕"}
        colors = {"Elliptical": WARNING, "Spiral": ACCENT, "Bar-Spiral": ACCENT2}
        col = colors.get(label, TEXT_PRIMARY)
        self.lbl_icon.config(text=icons.get(label, "?"), fg=col)
        self.lbl_pred.config(text=label, fg=col)
        self.lbl_conf.config(text=f"Confidence: {conf*100:.1f}%")

        # Probability bars
        self._draw_bars(probs, colors)

        # Description
        descs = {
            "Elliptical":
                "Smooth, featureless galaxies with an ellipsoidal shape. "
                "They contain mostly older red stars and little gas or dust, "
                "resulting in minimal star formation activity.",
            "Spiral":
                "Disk-shaped galaxies with prominent spiral arms winding outward "
                "from a central bulge. They contain young blue stars, gas, and dust "
                "along their arms where active star formation occurs.",
            "Bar-Spiral":
                "Similar to spiral galaxies but with a central elongated bar of "
                "stars from which the spiral arms extend. The bar channels gas "
                "toward the nucleus, fuelling star formation.",
        }
        self.lbl_desc.config(text=descs.get(label, ""))

    def _draw_bars(self, probs, colors):
        c = self.bar_canvas
        c.delete("all")
        bar_h = 28
        gap   = 14
        max_w = 360
        y     = 10
        default_colors = {"Elliptical": WARNING, "Spiral": ACCENT, "Bar-Spiral": ACCENT2}
        for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
            col = colors.get(name, default_colors.get(name, ACCENT))
            # background track
            c.create_rectangle(0, y, max_w, y+bar_h, fill=BG_CARD, outline="")
            # filled bar
            bar_w = max(4, int(prob * max_w))
            c.create_rectangle(0, y, bar_w, y+bar_h, fill=col, outline="")
            # label
            c.create_text(8, y + bar_h//2, text=name, anchor="w",
                          font=("Segoe UI", 10, "bold"), fill=BG_DARK if bar_w > 60 else TEXT_PRIMARY)
            # pct
            c.create_text(max_w - 4, y + bar_h//2, text=f"{prob*100:.1f}%",
                          anchor="e", font=FONT_SMALL, fill=TEXT_SEC)
            y += bar_h + gap

    def _go_back(self):
        self.app.show_page("UploadPage")

    def _save_result(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile="galaxy_result.txt"
        )
        if path:
            with open(path, "w") as f:
                f.write(f"Galaxy Classifier Result\n{'='*30}\n")
                f.write(f"Image   : {self.app.image_path.get()}\n")
                f.write(f"Label   : {self.app.result_label.get()}\n")
                f.write(f"Confidence: {self.app.result_conf.get()*100:.1f}%\n\n")
                f.write("All probabilities:\n")
                for k, v in self.app.all_probs.items():
                    f.write(f"  {k:12s}: {v*100:.1f}%\n")
            messagebox.showinfo("Saved", f"Result saved to:\n{path}")


# ─────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image, ImageTk
    
    # Check for required dependencies
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess, sys
        print("Installing openai for GPT-5.4 Vision validation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        import subprocess, sys
        print("Installing python-dotenv for environment variables...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv", "-q"])
    
    # Check if .env file exists
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Missing Configuration",
            ".env file not found!\n\n"
            "For validation to work, create a .env file in this directory with:\n"
            "GEMINI_KEY_1=your_api_key_here\n"
            "GEMINI_KEY_2=your_api_key_here\n"
            "(You can add up to 6 keys)\n\n"
            "Without this, validation will fail.\n"
            "The app will still run but validation won't work."
        )
        root.destroy()

    #testing 
    app = GalaxyApp()
    app.mainloop()
