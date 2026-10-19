"""Run a tiny CPU-only synthetic simulation."""
from pantheon import Simulation
from pantheon.hardware import detect_devices


if __name__ == "__main__":
    print("Hardware (presence does not imply backend support):")
    for device in detect_devices():
        print(f"  {device.backend}: {device.available} ({device.status})")
    result = Simulation(population=1000, seed=42).run(ticks=10)
    print(result)
