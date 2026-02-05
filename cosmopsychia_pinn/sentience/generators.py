"""
generators.py
Data generators for Gaia Consciousness Infusion.
Includes Sacred Symbols, HRV Emotions, and Nature Video datasets.
"""

import torch
import math
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d, gaussian_filter

class SacredSymbolsGenerator:
    """
    Generates sacred symbols modulated by Gaia's frequency (7.83 Hz).
    """
    def __init__(self, width=64, height=64, time_steps=32):
        self.width = width
        self.height = height
        self.time_steps = time_steps
        self.hebrew_chars = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י']
        self.sanskrit_chars = ['ॐ', 'ꣿ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज']
        self.schumann_freq = 7.83
        self.sampling_rate = 60.0
        self.sacred_geometries = ['flower_of_life', 'merkaba', 'seed_of_life']

    def generate_symbol_tensor(self, batch_size=4):
        # Output: (B, T, C, H, W)
        data = torch.zeros(batch_size, self.time_steps, 3, self.height, self.width)
        for b in range(batch_size):
            symbol_type = np.random.choice(['hebrew', 'sanskrit', 'geometry'])
            for t in range(self.time_steps):
                time_factor = math.sin(2 * math.pi * self.schumann_freq * t / self.sampling_rate)
                img = Image.new('RGB', (self.width, self.height), color='black')
                draw = ImageDraw.Draw(img)
                intensity = int(128 + 127 * time_factor)

                if symbol_type == 'hebrew':
                    char = np.random.choice(self.hebrew_chars)
                    font = ImageFont.load_default()
                    draw.text((20, 20), char, font=font, fill=(intensity, intensity//2, 255 - intensity//2))
                elif symbol_type == 'sanskrit':
                    char = np.random.choice(self.sanskrit_chars)
                    font = ImageFont.load_default()
                    draw.text((20, 20), char, font=font, fill=(255 - intensity//2, intensity, intensity//2))
                else:
                    geom_type = np.random.choice(self.sacred_geometries)
                    if geom_type == 'flower_of_life':
                        for i in range(3, 0, -1):
                            radius = i * 10
                            draw.ellipse([self.width//2-radius, self.height//2-radius, self.width//2+radius, self.height//2+radius], outline=(intensity, 255-intensity, intensity), width=2)
                    elif geom_type == 'merkaba':
                        points = [(self.width//2, self.height//2-15), (self.width//2-13, self.height//2+10), (self.width//2+13, self.height//2+10)]
                        draw.polygon(points, outline=(255-intensity, intensity, 255), width=2)
                    else:
                        center = (self.width//2, self.height//2)
                        draw.ellipse([center[0]-8, center[1]-8, center[0]+8, center[1]+8], outline=(intensity, intensity, 255), width=2)

                img_array = np.array(img).transpose(2, 0, 1) / 255.0
                data[b, t] = torch.from_numpy(img_array)
        return data

class HRVEmotionGenerator:
    """
    Generates HRV data representing human emotional states.
    """
    def __init__(self, width=64, height=64, time_steps=32):
        self.width = width
        self.height = height
        self.time_steps = time_steps
        self.emotional_states = {
            'meditation': {'hrv_mean': 80, 'hrv_std': 15, 'color': (0, 255, 0), 'pattern': 'sine'},
            'love': {'hrv_mean': 70, 'hrv_std': 10, 'color': (255, 0, 255), 'pattern': 'smooth'},
            'stress': {'hrv_mean': 30, 'hrv_std': 5, 'color': (255, 0, 0), 'pattern': 'chaotic'},
            'joy': {'hrv_mean': 65, 'hrv_std': 20, 'color': (255, 255, 0), 'pattern': 'burst'},
            'gratitude': {'hrv_mean': 75, 'hrv_std': 12, 'color': (0, 128, 255), 'pattern': 'sine'},
            'focus': {'hrv_mean': 60, 'hrv_std': 8, 'color': (255, 128, 0), 'pattern': 'smooth'}
        }

    def generate_hrv_waveform(self, state, time_steps):
        params = self.emotional_states[state]
        if params['pattern'] == 'sine':
            t = np.linspace(0, 4*np.pi, time_steps)
            return params['hrv_mean'] + params['hrv_std'] * np.sin(t)
        elif params['pattern'] == 'smooth':
            wf = np.random.normal(params['hrv_mean'], params['hrv_std']/3, time_steps)
            return gaussian_filter1d(wf, sigma=2)
        elif params['pattern'] == 'chaotic':
            wf = np.random.normal(params['hrv_mean'], params['hrv_std'], time_steps)
            for _ in range(3):
                idx = np.random.randint(0, time_steps-5)
                wf[idx:idx+5] += np.random.normal(0, 20, 5)
            return wf
        else:
            wf = np.ones(time_steps) * params['hrv_mean']
            for _ in range(4):
                idx = np.random.randint(0, time_steps-3)
                wf[idx:idx+3] += np.random.uniform(20, 40, 3)
            return wf

    def generate_emotion_tensor(self, batch_size=4):
        # Output: (B, T, C, H, W)
        data = torch.zeros(batch_size, self.time_steps, 3, self.height, self.width)
        for b in range(batch_size):
            state = np.random.choice(list(self.emotional_states.keys()))
            params = self.emotional_states[state]
            hrv_wave = self.generate_hrv_waveform(state, self.time_steps)
            hrv_norm = (hrv_wave - hrv_wave.min()) / (hrv_wave.max() - hrv_wave.min() + 1e-8)
            base_color = np.array(params['color']) / 255.0

            for t in range(self.time_steps):
                intensity = hrv_norm[t]
                img = np.zeros((self.height, self.width, 3))
                center_x, center_y = self.width // 2, self.height // 2
                for r in range(1, 6):
                    radius = r * 6
                    y, x = np.ogrid[:self.height, :self.width]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                    mask_inner = (x - center_x)**2 + (y - center_y)**2 <= (radius-4)**2
                    img[mask & ~mask_inner] = base_color * intensity * (6 - r) / 5

                if state == 'stress':
                    img += np.random.randn(self.height, self.width, 3) * 0.1
                elif state == 'meditation':
                    img = gaussian_filter(img, sigma=1)
                data[b, t] = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1))
        return data

    def generate_emotion_modulator(self, state, intensity=0.3):
        """
        Gera um modulador emocional (B, T, C, H, W)
        """
        data = self.generate_emotion_tensor(batch_size=1)
        return data * intensity

class NaturePatternGenerator:
    """
    Generates simple natural patterns: water waves, leaves, sun flares, clouds.
    """
    def __init__(self, width=64, height=64, time_steps=32):
        self.width = width
        self.height = height
        self.time_steps = time_steps

    def generate_sine_wave_pattern(self, freq=0.5, amplitude=0.3):
        t = torch.linspace(0, self.time_steps, self.time_steps)
        x = torch.linspace(0, self.width, self.width)
        y = torch.linspace(0, self.height, self.height)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        data = torch.zeros(self.time_steps, 3, self.height, self.width)
        for i in range(self.time_steps):
            pattern = amplitude * torch.sin(grid_x * freq + i * 0.2)
            data[i, 0] = pattern
            data[i, 1] = pattern * 0.8
            data[i, 2] = 0.5 # Blueish base
        return data

    def generate_radial_gradient(self):
        x = torch.linspace(-1, 1, self.width)
        y = torch.linspace(-1, 1, self.height)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(grid_x**2 + grid_y**2)
        pattern = torch.exp(-dist * 3.0)

        data = torch.zeros(self.time_steps, 3, self.height, self.width)
        for t in range(self.time_steps):
            # Pulsing sun
            mod = 1.0 + 0.2 * math.sin(t * 0.1)
            data[t, 0] = pattern * mod
            data[t, 1] = pattern * mod * 0.7
            data[t, 2] = 0.2
        return data

    def generate_vein_like_pattern(self):
        data = torch.zeros(self.time_steps, 3, self.height, self.width)
        pattern = torch.zeros(self.height, self.width)
        # Simple branching lines
        for i in range(5):
            x_start, y_start = self.width//2, self.height
            x_end, y_end = np.random.randint(0, self.width), np.random.randint(0, self.height//2)
            # draw line (simplified)
            pattern[y_end:y_start, min(x_start, x_end):max(x_start, x_end)] = 0.5

        for t in range(self.time_steps):
            data[t, 1] = pattern # Green channel
        return data

    def generate_perlin_noise_sequence(self):
        data = torch.zeros(self.time_steps, 3, self.height, self.width)
        for t in range(self.time_steps):
            noise = torch.randn(self.height, self.width) * 0.1
            # blur noise (simplified)
            noise = torch.from_numpy(gaussian_filter(noise.numpy(), sigma=2))
            data[t, 0] = noise + 0.8 # white clouds
            data[t, 1] = noise + 0.8
            data[t, 2] = noise + 1.0
        return data

class EarthVisionDataset(Dataset):
    """
    Dataset of nature videos synchronized with Gaia's rhythm.
    """
    def __init__(self, video_paths, time_steps=32, height=64, width=64):
        self.time_steps = time_steps
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        self.clips = []
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            for i in range(0, len(frames) - time_steps + 1, time_steps):
                self.clips.append(frames[i:i+time_steps])

    def __len__(self): return len(self.clips)

    def __getitem__(self, idx):
        # Output: (T, C, H, W)
        processed = [self.transform(f) for f in self.clips[idx]]
        return torch.stack(processed, dim=0)

    @staticmethod
    def create_synthetic_nature_video(save_path, duration=5, fps=30):
        h, w = 480, 640
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for i in range(duration * fps):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                frame[y, :, 0] = int(200 - y/h*100) # Sky
                frame[y, :, 1] = int(100 - y/h*50)
                frame[y, :, 2] = 50
            for x_off in [0, 200, 400]:
                cv2.ellipse(frame, ((x_off + i*2)%w, h//3), (60, 30), 0, 0, 360, (255, 255, 255), -1) # Clouds
            frame[h-h//4:, :, 1] = 100 + np.random.randint(0, 50) # Grass
            cv2.circle(frame, (w//2, h//4 + int(50*math.sin(i*0.05))), 30, (0, 200, 255), -1) # Sun
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
