import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import numpy as np
import os


class HeatmapGenerator:
    def __init__(self, pitch_image_path="../assets/generated_pitch.png"):
        self.pitch_image_path = pitch_image_path

        # Słownik na dane. Kluczami będą ID z In2Teams (1 i 2)
        self.team_data = {}

        # Ładowanie tła
        if os.path.exists(pitch_image_path):
            self.bg_img = cv2.imread(pitch_image_path)
            # convert BGR->RGB dla plt
            self.bg_img = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2RGB)
            # poprawne przypisanie wymiarów (width = cols, height = rows)
            self.height = self.bg_img.shape[0]
            self.width = self.bg_img.shape[1]
        else:
            self.bg_img = None
            # wartości domyślne w px
            self.width = 1200
            self.height = 700

        print(f"[HeatmapGenerator] Background loaded: {self.pitch_image_path}, size: {self.width}x{self.height}")

    def update(self, transformed_players):
        """
        transformed_players: lista krotek [(x, y, team_id), ...]
        """
        if not transformed_players:
            # nic nie dodajemy, ale logujemy
            print("[HeatmapGenerator] update called with 0 transformed players")
            return

        for x, y, team_id in transformed_players:
            if team_id not in self.team_data:
                self.team_data[team_id] = []
            # upewnij się, że x,y są floatami
            self.team_data[team_id].append([float(x), float(y)])

    def save_heatmaps(self, output_dir="output_heatmaps"):
        """Generuje oddzielne heatmapy dla Team 1 i Team 2"""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not self.team_data:
            print("[HeatmapGenerator] Brak zebranych punktów dla żadnej drużyny - nic do zapisania.")
            return

        # --- KONFIGURACJA ZGODNA Z TWOIM KODEM ANALYZE ---
        team_configs = {
            1: {"cmap": "Blues", "name": "Team_1_Blue"},
            2: {"cmap": "Reds", "name": "Team_2_Red"}
        }

        # Iterujemy po zebranych danych
        for team_id, data in self.team_data.items():
            config = team_configs.get(team_id, {"cmap": "Greens", "name": f"Team_{team_id}_Unknown"})
            print(f"[HeatmapGenerator] Generowanie heatmapy dla: {config['name']} (Liczba punktów: {len(data)})")

            output_filename = os.path.join(output_dir, f"{config['name']}_heatmap.png")
            self._generate_single_heatmap(data, config['cmap'], output_filename)

    def _generate_single_heatmap(self, data, cmap, filename):
        if not data or len(data) == 0:
            print(f"[HeatmapGenerator] _generate_single_heatmap: brak danych dla {filename}")
            return

        df = pd.DataFrame(data, columns=['x', 'y'])

        # Dodaj drobny jitter, żeby uniknąć singular matrix w KDE (jeśli wszystkie punkty na tej samej pozycji)
        if len(df) <= 50:
            # tylko niewielka losowość
            df['x'] = df['x'] + np.random.normal(0, 0.5, len(df))
            df['y'] = df['y'] + np.random.normal(0, 0.5, len(df))

        plt.figure(figsize=(self.width / 100.0, self.height / 100.0))

        # Tło
        if self.bg_img is not None:
            # extent = [xmin, xmax, ymin, ymax] w kolejności x then y (wysokość odwrócona)
            plt.imshow(self.bg_img, extent=[0, self.width, self.height, 0], aspect='auto')

        try:
            sns.kdeplot(
                data=df,
                x='x',
                y='y',
                fill=True,
                alpha=0.6,
                cmap=cmap,
                thresh=0.05,
                levels=15,
                bw_adjust=0.7,
                clip=((0, self.width), (0, self.height))
            )
        except Exception as e:
            print(f"[HeatmapGenerator] Błąd generowania KDE dla {filename}: {e}")
            plt.close()
            return

        plt.xlim(0, self.width)
        plt.ylim(self.height, 0)
        plt.axis('off')

        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        print(f"[HeatmapGenerator] Zapisano: {filename}")
