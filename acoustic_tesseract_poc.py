import numpy as np
import time
from sklearn import datasets
from sklearn.manifold import TSNE
from scipy.spatial import KDTree
import pygame
import math
import sys
from collections import deque

# --- CONFIGURATION ---
SOUND_FILE = 'sound.wav'
NUM_NEIGHBORS_TO_HEAR = 3
MOVE_SPEED = 2.5
TURN_SPEED = 0.08
MAX_HEARING_DISTANCE = 80.0
PING_INTERVAL = 0.8  # Reduced for more responsive audio
VOICE_ANNOUNCE_INTERVAL = 3.0

# --- VISUAL CONFIG ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
COLOR_BLACK = (5, 5, 15)
COLOR_RED = (255, 60, 60)
COLOR_WHITE = (255, 255, 255)
COLOR_UI_BG = (20, 20, 35, 200)
COLOR_UI_TEXT = (180, 220, 255)

CLUSTER_COLORS = [
    (0, 255, 128),
    (64, 156, 255),
    (255, 200, 0),
    (255, 64, 255),
    (255, 128, 0),
    (128, 255, 255),
    (200, 100, 255),
]

class Particle:
    def __init__(self, pos, vel, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifetime = lifetime
        self.age = 0
        
    def update(self, dt):
        self.pos += self.vel * dt
        self.age += dt
        self.vel *= 0.95
        
    def is_alive(self):
        return self.age < self.lifetime
        
    def get_alpha(self):
        return int(255 * (1 - self.age / self.lifetime))

class ParticleSystem:
    def __init__(self, max_particles=500):
        self.particles = []
        self.max_particles = max_particles
        
    def emit(self, pos, color, count=5):
        # Limit particle count to prevent memory issues
        if len(self.particles) + count > self.max_particles:
            self.particles = self.particles[-(self.max_particles - count):]
            
        for _ in range(count):
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(20, 60)
            vel = [math.cos(angle)*speed, math.sin(angle)*speed]
            self.particles.append(Particle(pos, vel, color, np.random.uniform(0.3, 0.8)))
            
    def update(self, dt):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update(dt)
            
    def draw(self, surface):
        for p in self.particles:
            alpha = p.get_alpha()
            color = (*p.color, alpha)
            surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (4, 4), 4)
            surface.blit(surf, (int(p.pos[0])-4, int(p.pos[1])-4))

class Trail:
    def __init__(self, max_length=30):
        self.points = deque(maxlen=max_length)
        
    def add(self, pos):
        self.points.append(pos.copy())
        
    def get_points(self):
        return list(self.points)

class AudioVisualizer:
    def __init__(self):
        self.active_sounds = {}
        
    def register_sound(self, idx, intensity):
        self.active_sounds[idx] = (intensity, time.time())
        
    def update(self):
        current = time.time()
        self.active_sounds = {k: v for k, v in self.active_sounds.items() 
                            if current - v[1] < 0.5}
        
    def get_intensity(self, idx):
        if idx in self.active_sounds:
            age = time.time() - self.active_sounds[idx][1]
            return self.active_sounds[idx][0] * (1 - age/0.5)
        return 0

class CoordinateMapper:
    """Efficient coordinate mapping that caches transformations"""
    def __init__(self):
        self.data_bounds = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.z_range = (0, 1)
        
    def compute_bounds(self, data_points):
        """Pre-compute mapping parameters"""
        min_x, min_y = data_points[:, :2].min(axis=0)
        max_x, max_y = data_points[:, :2].max(axis=0)
        
        dx = max_x - min_x or 1
        dy = max_y - min_y or 1
        
        self.scale_x = (SCREEN_WIDTH * 0.85) / dx
        self.scale_y = (SCREEN_HEIGHT * 0.85) / dy
        self.offset_x = SCREEN_WIDTH * 0.075 - min_x * self.scale_x
        self.offset_y = SCREEN_HEIGHT * 0.075 - min_y * self.scale_y
        
        self.z_range = (data_points[:,2].min(), data_points[:,2].max())
        
    def map_point(self, point):
        """Fast point mapping using pre-computed parameters"""
        sx = int(point[0] * self.scale_x + self.offset_x)
        sy = int(point[1] * self.scale_y + self.offset_y)
        size = int(np.interp(point[2], self.z_range, [3, 10]))
        return (sx, sy, size)
    
    def map_point_2d(self, point):
        """Map 2D point (for trail)"""
        sx = int(point[0] * self.scale_x + self.offset_x)
        sy = int(point[1] * self.scale_y + self.offset_y)
        return (sx, sy)

def load_data():
    choice = input("Use (I)ris dataset, (R)andom data, or (D)igits dataset? [I/r/d]: ").strip().lower()
    if choice == 'r':
        print("Generating random 150-point, 5D dataset...")
        X = np.random.rand(150, 5)
        y = np.random.randint(0, 7, 150)
    elif choice == 'd':
        print("Loading Digits dataset...")
        digits = datasets.load_digits()
        X, y = digits.data, digits.target
    else:
        print("Loading Iris dataset...")
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
    return X, y

def prepare_data_and_map(X, y):
    print("Running t-SNE (this may take a moment)...")
    perp = min(30, len(X)-1)
    tsne_model = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perp, n_jobs=-1)
    X_3D = tsne_model.fit_transform(X)
    print("Building KD-Tree for fast spatial queries...")
    kdtree = KDTree(X_3D)
    return kdtree, X_3D, y

def setup_audio():
    # Stereo output is critical for 3D audio
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(NUM_NEIGHBORS_TO_HEAR + 2)
    pygame.mixer.set_reserved(1)
    
    try:
        base_sound = pygame.mixer.Sound(SOUND_FILE)
    except:
        print(f"Warning: Could not load {SOUND_FILE}, creating harmonic tones")
        base_sound = create_beep_sound(261.63)
    
    # Convert to numpy array for stereo manipulation
    sound_array = pygame.sndarray.array(base_sound)
    return sound_array

def create_spatial_sound(base_sound_array, left_vol, right_vol):
    """Create a stereo sound with TRUE stereo separation"""
    # Handle both mono and stereo source sounds
    if len(base_sound_array.shape) == 1:
        # Mono source
        mono = base_sound_array.astype(np.float32)
    else:
        # Stereo source - mix to mono first
        mono = base_sound_array.mean(axis=1).astype(np.float32)
    
    # Create TRUE stereo with independent volumes
    # Multiply by float volumes first, then convert to int16
    left_channel = (mono * left_vol).clip(-32768, 32767).astype(np.int16)
    right_channel = (mono * right_vol).clip(-32768, 32767).astype(np.int16)
    
    stereo = np.column_stack((left_channel, right_channel))
    return pygame.sndarray.make_sound(stereo)
    """Create a harmonic tone"""
    sample_rate = 44100
    duration = 0.3
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    
    fundamental = np.sin(2 * np.pi * frequency * t)
    wave = fundamental / np.max(np.abs(fundamental))
    
    envelope = np.ones_like(wave)
    fade_samples = int(0.1 * sample_rate)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    wave = wave * envelope
    wave = (wave * 15000).astype(np.int16)
    # Return mono - we'll create stereo dynamically
    return wave

def draw_glow(surface, pos, radius, color, intensity=1.0):
    """Draw a glowing circle"""
    for i in range(3):
        r = radius + i*4
        alpha = int(80 * intensity / (i+1))
        surf = pygame.Surface((r*2+10, r*2+10), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, alpha), (r+5, r+5), r)
        surface.blit(surf, (pos[0]-r-5, pos[1]-r-5))

def draw_ui(surface, font, small_font, user_pos, nearest_dist, cluster_count, speed, show_help, fps):
    """Draw UI overlay with FPS counter"""
    panel = pygame.Surface((SCREEN_WIDTH, 120), pygame.SRCALPHA)
    pygame.draw.rect(panel, COLOR_UI_BG, (0, 0, SCREEN_WIDTH, 120), border_radius=0)
    surface.blit(panel, (0, 0))
    
    title = font.render("ACOUSTIC TESSERACT", True, (100, 200, 255))
    surface.blit(title, (20, 15))
    
    stats = [
        f"Position: ({user_pos[0]:.1f}, {user_pos[1]:.1f}, {user_pos[2]:.1f}) | FPS: {fps:.0f}",
        f"Nearest: {nearest_dist:.1f}m | Clusters: {cluster_count} | Speed: {speed:.1f}",
    ]
    
    for i, stat in enumerate(stats):
        text = small_font.render(stat, True, COLOR_UI_TEXT)
        surface.blit(text, (20, 55 + i*20))
    
    if show_help:
        help_panel = pygame.Surface((400, 220), pygame.SRCALPHA)
        pygame.draw.rect(help_panel, (10, 10, 20, 230), (0, 0, 400, 220), border_radius=10)
        surface.blit(help_panel, (SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 - 110))
        
        controls = [
            "CONTROLS:",
            "W/S - Move Forward/Backward",
            "A/D or Arrows - Turn Left/Right",
            "UP Arrow - Boost Speed",
            "SPACE - Reset Position",
            "H - Toggle Help",
            "Q/ESC - Quit"
        ]
        for i, line in enumerate(controls):
            text = small_font.render(line, True, (255, 255, 255))
            surface.blit(text, (SCREEN_WIDTH//2 - 180, SCREEN_HEIGHT//2 - 90 + i*25))
    else:
        hint = small_font.render("Press H for Help", True, (150, 150, 150))
        surface.blit(hint, (SCREEN_WIDTH - 180, 20))

def draw_direction_indicator(surface, pos, direction, length=40):
    """Draw direction arrow"""
    end_x = pos[0] + direction[0] * length
    end_y = pos[1] + direction[1] * length
    
    pygame.draw.line(surface, (255, 100, 100), pos, (end_x, end_y), 3)
    
    angle = math.atan2(direction[1], direction[0])
    arrow_size = 10
    left = (end_x - arrow_size*math.cos(angle-0.5), end_y - arrow_size*math.sin(angle-0.5))
    right = (end_x - arrow_size*math.cos(angle+0.5), end_y - arrow_size*math.sin(angle+0.5))
    pygame.draw.polygon(surface, (255, 100, 100), [(end_x, end_y), left, right])

def draw_radar(surface, user_pos, data_points, labels, user_scr, max_dist=100):
    """Draw mini-radar"""
    radar_size = 150
    radar_pos = (SCREEN_WIDTH - radar_size - 20, SCREEN_HEIGHT - radar_size - 20)
    
    radar_surf = pygame.Surface((radar_size, radar_size), pygame.SRCALPHA)
    pygame.draw.circle(radar_surf, (20, 40, 60, 200), (radar_size//2, radar_size//2), radar_size//2)
    pygame.draw.circle(radar_surf, (40, 80, 120, 100), (radar_size//2, radar_size//2), radar_size//2, 2)
    
    for i, point in enumerate(data_points):
        dist = np.linalg.norm(point - user_pos)
        if dist < max_dist:
            rel_pos = point - user_pos
            radar_x = int(radar_size//2 + (rel_pos[0]/max_dist) * (radar_size//2))
            radar_y = int(radar_size//2 + (rel_pos[1]/max_dist) * (radar_size//2))
            if 0 <= radar_x < radar_size and 0 <= radar_y < radar_size:
                color = CLUSTER_COLORS[labels[i] % len(CLUSTER_COLORS)]
                pygame.draw.circle(radar_surf, color, (radar_x, radar_y), 2)
    
    pygame.draw.circle(radar_surf, (255, 60, 60), (radar_size//2, radar_size//2), 4)
    surface.blit(radar_surf, radar_pos)

def main():
    X, labels = load_data()
    kdtree, data_points, labels = prepare_data_and_map(X, labels)

    pygame.init()
    base_sound_array = setup_audio()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enhanced Acoustic Tesseract - Navigate with Sound")
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.Font(None, 42)
        small_font = pygame.font.Font(None, 24)
    except:
        font = pygame.font.SysFont('arial', 42)
        small_font = pygame.font.SysFont('arial', 24)

    # Initialize coordinate mapper
    mapper = CoordinateMapper()
    mapper.compute_bounds(data_points)
    
    # Pre-scale all data points once
    scaled_points = [mapper.map_point(p) for p in data_points]
    
    user_pos = np.mean(data_points, axis=0)
    initial_pos = user_pos.copy()
    user_dir = np.array([1.0, 0.0, 0.0])
    
    particles = ParticleSystem(max_particles=500)
    trail = Trail(50)
    visualizer = AudioVisualizer()
    last_ping_time = {}
    
    show_help = False
    boost_active = False
    frame_count = 0
    
    print("\n=== ACOUSTIC TESSERACT ENHANCED ===")
    print("Navigate through sound! Each data point emits audio.")
    print("Press H for controls | SPACE to reset position")
    print("Accessibility: Spatial audio with proximity alerts")
    print("=====================================\n")

    running = True
    while running:
        dt = clock.tick(60)/1000.0
        frame_count += 1
        fps = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_SPACE:
                    user_pos = initial_pos.copy()
                    user_dir = np.array([1.0, 0.0, 0.0])
                    trail.points.clear()

        keys = pygame.key.get_pressed()
        
        # Movement
        move_delta = MOVE_SPEED * dt * 30
        if keys[pygame.K_w]: 
            user_pos += user_dir * move_delta
            if frame_count % 3 == 0:
                trail.add(user_pos[:2])
        if keys[pygame.K_s]: 
            user_pos -= user_dir * move_delta
            if frame_count % 3 == 0:
                trail.add(user_pos[:2])
        
        # Speed boost
        if keys[pygame.K_UP]:
            user_pos += user_dir * move_delta * 1.8
            boost_active = True
            if frame_count % 3 == 0:
                trail.add(user_pos[:2])
        else:
            boost_active = False
                
        # Rotation
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            angle = TURN_SPEED
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            user_dir = R @ user_dir
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            angle = -TURN_SPEED
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            user_dir = R @ user_dir

        # Spatial audio
        try:
            distances, indices = kdtree.query(user_pos, k=NUM_NEIGHBORS_TO_HEAR)
            if np.isscalar(distances): 
                distances = [distances]
                indices = [indices]
        except:
            distances = []
            indices = []

        current_time = time.time()
        nearest_dist = distances[0] if len(distances) > 0 else 999

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            try:
                if dist > MAX_HEARING_DISTANCE: 
                    continue
                    
                if idx not in last_ping_time:
                    last_ping_time[idx] = 0
                    
                if current_time - last_ping_time[idx] < PING_INTERVAL:
                    continue

                last_ping_time[idx] = current_time

                point = data_points[idx]
                
                # Calculate base volume (distance-based)
                base_vol = max(0, 1-(dist/MAX_HEARING_DISTANCE)) ** 1.5
                base_vol *= 0.8  # Master volume
                
                # Calculate stereo pan based on direction
                vec = point - user_pos
                vec_norm = np.linalg.norm(vec[:2])  # Use 2D distance for panning
                
                if vec_norm > 1e-6:
                    # Right vector (perpendicular to forward direction)
                    right = np.array([-user_dir[1], user_dir[0], 0])
                    
                    # Dot product gives us left-right position (-1 = left, +1 = right)
                    pan = np.dot(vec, right) / vec_norm
                    pan = np.clip(pan, -1, 1)
                    
                    # Forward vector (how much in front vs behind)
                    forward_amount = np.dot(vec, user_dir) / vec_norm
                    
                    # Reduce volume for sounds behind us
                    if forward_amount < 0:
                        base_vol *= 0.2  # Sounds behind are much quieter
                else:
                    pan = 0
                
                # TRUE STEREO: Convert pan to extreme stereo separation
                # pan = -1: LEFT EAR ONLY (left=1.0, right=0.0)
                # pan = 0:  BOTH EARS (left=0.7, right=0.7)
                # pan = +1: RIGHT EAR ONLY (left=0.0, right=1.0)
                
                # Use power function for sharper stereo separation
                left_vol = base_vol * (1.0 - max(0, pan)) ** 0.5
                right_vol = base_vol * (1.0 + min(0, pan)) ** 0.5
                
                # Ensure at least some signal for debugging
                # But create REAL stereo difference
                if pan < -0.3:  # Sound is LEFT
                    right_vol *= 0.1  # Almost silent in right ear
                elif pan > 0.3:  # Sound is RIGHT  
                    left_vol *= 0.1  # Almost silent in left ear
                
                # Create spatial sound with EXTREME stereo separation
                spatial_sound = create_spatial_sound(base_sound_array, left_vol, right_vol)
                
                # Play on available channel
                chan = pygame.mixer.Channel(i % NUM_NEIGHBORS_TO_HEAR)
                chan.play(spatial_sound)
                
                visualizer.register_sound(idx, base_vol)
                
                # Debug print for testing
                if frame_count % 60 == 0 and i == 0:  # Print once per second for nearest sound
                    print(f"Sound {idx}: pan={pan:.2f}, L={left_vol:.2f}, R={right_vol:.2f}")
                    
            except Exception as e:
                print(f"Audio error: {e}")
                continue

        particles.update(dt)
        visualizer.update()

        # Render
        screen.fill(COLOR_BLACK)
        
        # Grid
        for i in range(0, SCREEN_WIDTH, 50):
            pygame.draw.line(screen, (15, 15, 25), (i, 0), (i, SCREEN_HEIGHT), 1)
        for i in range(0, SCREEN_HEIGHT, 50):
            pygame.draw.line(screen, (15, 15, 25), (0, i), (SCREEN_WIDTH, i), 1)
        
        user_scr = mapper.map_point_2d(user_pos[:2])
        
        if boost_active and frame_count % 5 == 0:
            particles.emit(user_scr, (255, 100, 50), 3)
        
        # Audio connection lines
        for i, (dist, idx) in enumerate(zip(distances[:5], indices[:5])):
            if dist < MAX_HEARING_DISTANCE:
                intensity = visualizer.get_intensity(idx)
                if intensity > 0:
                    alpha = int(intensity * 100)
                    color = (*CLUSTER_COLORS[labels[idx] % len(CLUSTER_COLORS)], alpha)
                    surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(surf, color, user_scr, scaled_points[idx][:2], 2)
                    screen.blit(surf, (0, 0))
        
        # Efficient trail rendering
        trail_points = trail.get_points()
        trail_scaled = [mapper.map_point_2d(p) for p in trail_points]
        
        for i in range(len(trail_scaled)-1):
            alpha = int(255 * (i / max(len(trail_scaled), 1)))
            color = (255, 60, 60, min(alpha, 255))
            pygame.draw.line(screen, color, trail_scaled[i], trail_scaled[i+1], 2)
        
        # Data points
        for i, (x, y, s) in enumerate(scaled_points):
            color = CLUSTER_COLORS[labels[i] % len(CLUSTER_COLORS)]
            intensity = visualizer.get_intensity(i)
            if intensity > 0.1:
                draw_glow(screen, (x, y), s+4, color, intensity)
            pygame.draw.circle(screen, color, (x, y), s)
            
        particles.draw(screen)
        
        # Player
        draw_glow(screen, user_scr, 10, COLOR_RED, 1.0)
        pygame.draw.circle(screen, COLOR_RED, user_scr, 8)
        
        dir_2d = user_dir[:2] / (np.linalg.norm(user_dir[:2]) + 1e-9)
        draw_direction_indicator(screen, user_scr, dir_2d, 50 if boost_active else 30)
        
        draw_radar(screen, user_pos, data_points, labels, user_scr)
        
        cluster_count = len(set(labels))
        draw_ui(screen, font, small_font, user_pos, nearest_dist, cluster_count, 
                MOVE_SPEED, show_help, fps)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()