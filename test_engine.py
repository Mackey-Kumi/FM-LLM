"""Quick smoke test of the GridForecast engine: python test_engine.py"""

from app.inference import ForecastEngine

print("Testing ForecastEngine...")
engine = ForecastEngine(device="cpu")
print(f"Checkpoints: {engine.available_checkpoints()} (live: {engine.live_checkpoints()})")
print(f"Channels: {engine.get_channel_names()}")

checkpoint = engine.available_checkpoints()[0]
window = engine.windows(checkpoint)[0]
result = engine.forecast(window, 192, checkpoint)
print(f"Forecast from {result['dates'][0]} ({result['source']}): shape {result['predictions'].shape}")
print(f"MSE={result['metrics']['mse']:.4f}  MAE={result['metrics']['mae']:.4f} (normalized)")
print("ForecastEngine works!")
