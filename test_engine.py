import os
from dotenv import load_dotenv
load_dotenv()

import torch
from app.inference import ForecastEngine

print('Testing ForecastEngine...')
engine = ForecastEngine(checkpoint_name='instnorm', device='cpu')
print(f'Available windows: {engine.get_available_windows()}')
print(f'Channels: {engine.get_channel_names()}')
result = engine.forecast_window(0, 192)
print(f'Forecast shape: {result["predictions"].shape}')
mse = ((result["predictions"] - result["ground_truth"]) ** 2).mean()
print(f'MSE: {mse:.4f}')
print('ForecastEngine works!')