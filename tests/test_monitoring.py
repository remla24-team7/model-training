import time
import psutil
from scripts.train import load_data, train_model, build_model


# Function to measure RAM usage
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  # Convert to MB


# Pytest test function
def test_model_performance(params):
    model = build_model(params)
    x_train, y_train, x_val, y_val = load_data(params)
    # Measure initial RAM usage
    initial_ram = get_ram_usage()

    # Measure training speed
    start_time = time.time()
    _ = train_model(model, params, x_train[:10], y_train[:10], validation_data=(x_val[:10], y_val[:10]))
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Measure RAM usage after training
    post_train_ram = get_ram_usage()
    print(f"Initial RAM usage: {initial_ram:.2f} MB")
    print(f"Post-training RAM usage: {post_train_ram:.2f} MB")

    # Assert statements to check for regressions (replace with your own thresholds)
    assert training_time < 60, "Training time is too high!"
    assert (post_train_ram - initial_ram) < 100, "Memory usage increased significantly!"
