#!/usr/bin/env python3
"""Dựng video Phát thanh Măng non có lời đọc từ script.txt + assets/background.jpg.

Cần: piper-tts, numpy, soundfile, pillow, ffmpeg và mô hình giọng
vi_VN-vais1000-medium (onnx + json) – đường dẫn đặt qua biến VOICE_DIR.
"""
import math
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.environ.get("VOICE_DIR", os.path.join(ROOT, "voice"))
VOICE = os.path.join(VOICE_DIR, "vi_VN-vais1000-medium.onnx")
OUT_DIR = os.path.join(ROOT, "output")
BUILD = os.path.join(ROOT, "build")
W, H, FPS = 1920, 1080, 25
SR = 44100

# Khoảng nghỉ (giây)
INTRO = 5.0          # nhạc hiệu trước lời chào
GAP_LINE = 0.5      # giữa các câu
GAP_PARA = 1.1       # giữa các đoạn
GAP_SECTION = 2.2    # giữa các phần (nhạc nổi lên)
OUTRO = 8.0          # nhạc kết

# Giọng đọc
PITCH_UP = 1.5       # nửa cung, nâng tông cho trẻ trung
PACE = 1.12          # >1 = chậm hơn tốc độ gốc của mô hình

FONT_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_XB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_SB = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
BULLETS = ("🌸", "⭐", "💛")


# ---------------------------------------------------------------- script
def parse_script(path):
    items = []  # dict(kind=line|para|section, ...)
    section = ""
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("# ") or line == "#" or line.startswith("#   "):
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            items.append({"kind": "section", "title": section})
            continue
        if not line.strip():
            items.append({"kind": "para"})
            continue
        if "||" in line:
            disp, spoken = [s.strip() for s in line.split("||", 1)]
        else:
            disp = spoken = line.strip()
        bullet = False
        for b in BULLETS:
            if disp.startswith(b):
                disp, bullet = disp[len(b):].strip(), True
        items.append({"kind": "line", "text": disp, "spoken": spoken,
                      "bullet": bullet, "section": section})
    return items


# ---------------------------------------------------------------- TTS
def synth_all(items):
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(VOICE)
    # Giọng trẻ trung: nâng tông PITCH_UP nửa cung bằng cách đọc chậm hơn rồi phát nhanh lại
    # (tông và âm sắc cùng sáng lên, tốc độ cuối cùng giữ ≈ PACE).
    r = 2 ** (PITCH_UP / 12)
    vsr = voice.config.sample_rate
    for it in items:
        if it["kind"] != "line":
            continue
        text = it["spoken"]
        pace = PACE * (0.94 if text.endswith("!") else 1.0)  # câu cảm thán đọc tươi hơn
        cfg = SynthesisConfig(length_scale=pace * r, noise_scale=0.78, noise_w_scale=0.9)
        chunks = list(voice.synthesize(text, syn_config=cfg))
        pause = np.zeros(int(0.28 * vsr * r), dtype=np.float32)
        parts = []
        for c in chunks:
            x = c.audio_float_array
            nz = np.where(np.abs(x) > 0.01)[0]
            parts += [x[max(0, nz[0] - 200): nz[-1] + 600] if len(nz) else x, pause]
        audio = np.concatenate(parts[:-1])
        # cắt khoảng lặng hai đầu
        idx = np.where(np.abs(audio) > 0.01)[0]
        audio = audio[max(0, idx[0] - 200): idx[-1] + 600] if len(idx) else audio
        # nâng tông + chuyển lên 44.1 kHz
        n = int(len(audio) * SR / (vsr * r))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        it["audio"] = audio.astype(np.float32)
        print(f"  tts {len(audio)/SR:5.1f}s  {it['spoken'][:60]}", flush=True)


def timeline(items):
    t = INTRO
    prev = None
    first_section = True
    for it in items:
        if it["kind"] == "section":
            if not first_section:
                t += GAP_SECTION
            first_section = False
            it["t"] = t
            prev = "section"
        elif it["kind"] == "para":
            if prev == "line":
                t += GAP_PARA - GAP_LINE
            prev = "para"
        else:
            it["start"] = t
            it["end"] = t + len(it["audio"]) / SR
            t = it["end"] + GAP_LINE
            prev = "line"
    return t + OUTRO


# ---------------------------------------------------------------- music
def note_freq(n):  # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


def pluck(freq, dur, sr=SR, decay=3.2):
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-decay * t) * np.minimum(1, t / 0.004)
    s = (np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-6 * t)
         + 0.12 * np.sin(2 * np.pi * 4.01 * freq * t) * np.exp(-12 * t))
    return s * env


def pad(freqs, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    env = np.minimum(1, t / 0.6) * np.minimum(1, (dur - t) / 0.6)
    s = sum(np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2.003 * f * t) for f in freqs)
    return s * env / len(freqs)


def make_music(total):
    bpm = 104
    beat = 60 / bpm
    # vòng hoà âm ngũ cung, vui tươi: C – Am – F – G | C – Em – F – G
    prog = [(48, [60, 64, 67]), (45, [57, 60, 64]), (41, [57, 60, 65]), (43, [55, 59, 62]),
            (48, [60, 64, 67]), (40, [55, 59, 64]), (41, [57, 60, 65]), (43, [55, 59, 62])]
    melody_bank = [  # giai điệu ngũ cung (C D E G A) cho mỗi ô nhịp, 8 móc đơn
        [72, 74, 76, 79, 76, 74, 72, 74], [76, 72, 69, 72, 74, 72, 69, 67],
        [69, 72, 74, 76, 74, 72, 69, 72], [74, 76, 79, 81, 79, 76, 74, 79],
        [79, 76, 74, 76, 72, 74, 76, 79], [76, 74, 71, 74, 76, 79, 76, 74],
        [72, 74, 76, 74, 72, 69, 72, 74], [74, 72, 74, 76, 79, 76, 74, 72],
    ]
    bar = 4 * beat
    n = int(total * SR) + SR
    mus = np.zeros(n, dtype=np.float64)

    def add(sig, t0, gain):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        if i < n:
            mus[i:j] += gain * sig[: j - i]

    bars = int(total / bar) + 2
    for b in range(bars):
        root, chord = prog[b % 8]
        t0 = b * bar
        add(pad([note_freq(x) for x in chord], bar + 0.5), t0, 0.18)
        for k in range(4):  # bass
            add(pluck(note_freq(root if k % 2 == 0 else root + 7), beat, decay=4), t0 + k * beat, 0.32)
        mel = melody_bank[b % 8]
        for k, m in enumerate(mel):  # giai điệu chuông
            if (b // 8) % 2 == 1 and k % 2 == 1:
                continue  # biến tấu nhẹ cho đỡ lặp
            add(pluck(note_freq(m), 1.2, decay=3.5), t0 + k * beat / 2, 0.22)
        for k in range(8):  # tiếng lắc nhẹ
            noise = np.random.default_rng(b * 8 + k).standard_normal(int(0.05 * SR))
            noise *= np.exp(-np.arange(len(noise)) / (0.008 * SR))
            add(np.diff(noise, prepend=0), t0 + k * beat / 2, 0.035 if k % 2 else 0.06)
    mus /= np.abs(mus).max()
    return mus[: int(total * SR)]


def music_envelope(items, total):
    """Nhạc to khi không có lời, hạ thấp (duck) khi cô đọc."""
    t = np.arange(int(total * SR)) / SR
    speaking = np.zeros_like(t, dtype=bool)
    for it in items:
        if it["kind"] == "line":
            speaking[int((it["start"] - 0.25) * SR): int((it["end"] + 0.1) * SR)] = True
    # khoảng nghỉ ngắn giữa câu vẫn giữ nhạc nhỏ
    target = np.where(speaking, 0.10, 0.10)
    for it in items:
        if it["kind"] == "section":
            a, b = it["t"] - GAP_SECTION + 0.2, it["t"] - 0.3
            target[(t > a) & (t < b)] = 0.42
    target[t < INTRO - 0.4] = 0.55
    last_end = max(it["end"] for it in items if it["kind"] == "line")
    target[t > last_end + 0.4] = 0.55
    # làm mượt (≈0.35 s)
    k = int(0.35 * SR)
    ker = np.ones(k) / k
    env = np.convolve(target, ker, mode="same")
    # fade in/out
    env *= np.minimum(1, t / 1.0) * np.clip((total - t) / 3.0, 0, 1)
    return env


# ---------------------------------------------------------------- graphics
def wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= maxw or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rounded(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def subtitle_layer(text, section, bullet):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_B, 52)
    maxw = 1560 - (60 if bullet else 0)
    lines = wrap(d, text, font, maxw)
    lh = 68
    box_h = len(lines) * lh + 52
    x0, x1 = 120, W - 120
    y1 = H - 48
    y0 = y1 - box_h
    # bóng đổ
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 4, y0 + 10, x1 + 4, y1 + 10), 30, fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(img)
    rounded(d, (x0, y0, x1, y1), 30, (13, 52, 140, 228))
    d.rounded_rectangle((x0, y0, x1, y1), radius=30, outline=(255, 214, 64, 255), width=4)
    ty = y0 + 26
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font)
        tx = (W - tw) / 2 + (30 if bullet else 0)
        if bullet and i == 0:
            cx, cy = tx - 42, ty + 32
            d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=(255, 214, 64, 255))
            d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(225, 40, 50, 255))
        d.text((tx + 2, ty + 3), ln, font=font, fill=(0, 0, 0, 120))
        d.text((tx, ty), ln, font=font, fill=(255, 255, 255, 255))
        ty += lh
    if section:
        sf_ = ImageFont.truetype(FONT_XB, 32)
        tw = d.textlength(section, font=sf_)
        bx0, by1 = x0 + 36, y0 + 4
        bx1, by0 = bx0 + tw + 56, by1 - 58
        rounded(d, (bx0, by0, bx1, by1 + 16), 22, (218, 37, 41, 255))
        d.rounded_rectangle((bx0, by0, bx1, by1 + 16), radius=22, outline=(255, 214, 64, 255), width=3)
        d.text((bx0 + 28, by0 + 11), section, font=sf_, fill=(255, 235, 120, 255))
    return img


def title_layer(alpha=1.0):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT_XB, 58)
    f2 = ImageFont.truetype(FONT_SB, 36)
    t1 = "PHÁT THANH MĂNG NON"
    t2 = "Hưởng ứng Ngày Văn hóa Việt Nam 24/11"
    t3 = "Người thực hiện: Cô giáo Nguyễn Thị Bích Hiền"
    w = max(d.textlength(t1, font=f1), d.textlength(t2, font=f2), d.textlength(t3, font=f2)) + 110
    x0, y1 = 120, H - 60
    y0 = y1 - 220
    rounded(d, (x0, y0, x0 + w, y1), 30, (13, 52, 140, 232))
    d.rectangle((x0, y0 + 20, x0 + 14, y1 - 20), fill=(218, 37, 41, 255))
    d.rounded_rectangle((x0, y0, x0 + w, y1), radius=30, outline=(255, 214, 64, 255), width=4)
    d.text((x0 + 55, y0 + 26), t1, font=f1, fill=(255, 214, 64, 255))
    d.text((x0 + 55, y0 + 104), t2, font=f2, fill=(255, 255, 255, 255))
    d.text((x0 + 55, y0 + 152), t3, font=f2, fill=(200, 225, 255, 255))
    return img


def end_layer():
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT_XB, 62)
    f2 = ImageFont.truetype(FONT_SB, 38)
    t1 = "Xin kính chào và hẹn gặp lại!"
    t2 = "Em yêu văn hóa Việt Nam – Tự hào bản sắc dân tộc"
    w = max(d.textlength(t1, font=f1), d.textlength(t2, font=f2)) + 120
    x0 = (W - w) / 2
    y1 = H - 60
    y0 = y1 - 180
    rounded(d, (x0, y0, x0 + w, y1), 30, (218, 37, 41, 235))
    d.rounded_rectangle((x0, y0, x0 + w, y1), radius=30, outline=(255, 214, 64, 255), width=4)
    d.text(((W - d.textlength(t1, font=f1)) / 2, y0 + 28), t1, font=f1, fill=(255, 225, 80, 255))
    d.text(((W - d.textlength(t2, font=f2)) / 2, y0 + 112), t2, font=f2, fill=(255, 255, 255, 255))
    return img


# ---------------------------------------------------------------- render
def render(items, total, audio_path, out_path):
    src = Image.open(os.path.join(ROOT, "assets", "background.jpg")).convert("RGB")
    # phóng nền lên 1.15 lần khung hình để có chỗ zoom/trượt
    base_w = int(W * 1.15)
    base_h = int(base_w * src.height / src.width)
    base = src.resize((base_w, base_h), Image.LANCZOS)

    # tiêu điểm camera cho từng phần: (x, y) tỉ lệ trên ảnh, mức zoom
    focus = {0: (0.50, 0.50, 1.00), 1: (0.30, 0.48, 1.16), 2: (0.70, 0.40, 1.16), 3: (0.68, 0.58, 1.14),
             4: (0.62, 0.62, 1.14), 5: (0.32, 0.48, 1.14), 6: (0.50, 0.48, 1.05)}
    sec_times = [0.0] + [it["t"] for it in items if it["kind"] == "section"]

    def cam(t):
        idx = max(i for i, st in enumerate(sec_times) if st <= t)
        ax, ay, az = focus[min(idx, 6)]
        prev = focus[max(min(idx, 6) - 1, 0)]
        # chuyển mượt 3 giây giữa các tiêu điểm
        k = min(1.0, (t - sec_times[idx]) / 3.0) if idx > 0 else 1.0
        k = k * k * (3 - 2 * k)
        x = prev[0] + (ax - prev[0]) * k
        y = prev[1] + (ay - prev[1]) * k
        z = prev[2] + (az - prev[2]) * k
        z *= 1 + 0.012 * math.sin(t * 2 * math.pi / 18)  # "thở" nhẹ
        return x, y, z

    lines = [it for it in items if it["kind"] == "line"]
    cache = {}
    title = title_layer()
    ending = end_layer()
    last_end = lines[-1]["end"]

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio_path,
           "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-x264-params", "aq-mode=3", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nframes = int(total * FPS)
    li = 0
    for f in range(nframes):
        t = f / FPS
        x, y, z = cam(t)
        cw, ch = base_w / z, base_w / z * H / W
        cw, ch = min(cw, base_w), min(ch, base_h)
        cx = min(max(x * base_w, cw / 2), base_w - cw / 2)
        cy = min(max(y * base_h, ch / 2), base_h - ch / 2)
        frame = base.resize((W, H), Image.BILINEAR,
                            box=(cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)).convert("RGBA")
        # lớp chữ
        while li < len(lines) - 1 and t >= lines[li + 1]["start"] - 0.05:
            li += 1
        cur = lines[li]
        overlay, alpha = None, 1.0
        if t < INTRO + 0.1 and t < cur["start"]:
            overlay = title
            alpha = min(1, max(0, (t - 0.6) / 0.6))
        elif t > last_end + 0.6:
            overlay = ending
            alpha = min(1, (t - last_end - 0.6) / 0.6)
        elif cur["start"] - 0.15 <= t <= cur["end"] + 0.45 or \
                (li + 1 < len(lines) and t < lines[li + 1]["start"] and t - cur["end"] < GAP_PARA + 0.1):
            key = id(cur)
            if key not in cache:
                cache.clear()
                cache[key] = subtitle_layer(cur["text"], cur["section"], cur["bullet"])
            overlay = cache[key]
            alpha = min(1, (t - cur["start"] + 0.15) / 0.15)
        elif t < INTRO + 0.1:
            overlay = title
        if overlay is not None:
            if alpha < 1:
                ov = overlay.copy()
                ov.putalpha(overlay.getchannel("A").point(lambda v: int(v * alpha)))
                overlay = ov
            frame = Image.alpha_composite(frame, overlay)
        proc.stdin.write(frame.convert("RGB").tobytes())
        if f % (FPS * 20) == 0:
            print(f"  render {t:6.1f}/{total:.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()


def fmt_srt(t):
    ms = int(round(t * 1000))
    return f"{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    items = parse_script(os.path.join(ROOT, "script.txt"))
    print("1) Tổng hợp giọng đọc…")
    synth_all(items)
    total = timeline(items)
    print(f"   thời lượng: {total/60:.1f} phút")

    print("2) Trộn âm thanh…")
    n = int(total * SR)
    voice = np.zeros(n, dtype=np.float32)
    for it in items:
        if it["kind"] == "line":
            i = int(it["start"] * SR)
            a = it["audio"] / max(1e-6, np.abs(it["audio"]).max()) * 0.9
            voice[i:i + len(a)] += a[: n - i]
    # màu giọng: lọc ù, tăng độ sáng/trong, thêm chút vang phòng thu
    vraw = os.path.join(BUILD, "voice_raw.wav")
    vfx = os.path.join(BUILD, "voice_fx.wav")
    sf.write(vraw, voice, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vraw, "-af",
                    "highpass=f=110,equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3200:t=q:w=1.2:g=3,"
                    "equalizer=f=9000:t=q:w=1:g=2,acompressor=threshold=-18dB:ratio=3:attack=5:release=80,"
                    "aecho=0.85:0.6:35|60:0.10|0.06", "-ar", str(SR), vfx], check=True)
    voice = sf.read(vfx, dtype="float32")[0][:n]
    voice = np.pad(voice, (0, n - len(voice)))
    music = make_music(total) * music_envelope(items, total)
    mix = voice + music.astype(np.float32)
    mix /= np.abs(mix).max() / 0.95
    stereo = np.stack([mix, mix], axis=1)
    raw_wav = os.path.join(BUILD, "mix_raw.wav")
    sf.write(raw_wav, stereo, SR)
    mix_wav = os.path.join(BUILD, "mix.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_wav,
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), mix_wav], check=True)

    print("3) Phụ đề SRT…")
    with open(os.path.join(OUT_DIR, "phu_de.srt"), "w", encoding="utf-8") as fh:
        k = 0
        for it in items:
            if it["kind"] == "line":
                k += 1
                fh.write(f"{k}\n{fmt_srt(it['start'])} --> {fmt_srt(it['end'] + 0.3)}\n{it['text']}\n\n")

    print("4) Dựng hình…")
    out = os.path.join(OUT_DIR, "phat_thanh_mang_non_van_hoa_viet_nam.mp4")
    render(items, total, mix_wav, out)
    print("Xong:", out)


if __name__ == "__main__":
    main()
