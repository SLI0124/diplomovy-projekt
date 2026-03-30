import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm


def download_dataset(url, output_path):
    if not os.path.exists(output_path):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(output_path, "wb") as file,
            tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
    else:
        print("Dataset already exists.")


def plot_global_temperature(input_file_path, save_file_path=None):
    data = pd.read_csv(input_file_path, delimiter=",", comment="#")

    required_columns = {"Year", "Anomaly"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Unexpected CSV format. Expected columns {required_columns}, got {set(data.columns)}"
        )

    font_size = 20

    data["5-Year Mean"] = data["Anomaly"].rolling(window=5).mean()

    plt.figure(figsize=(12, 8))
    plt.plot(
        data["Year"],
        data["Anomaly"],
        label="Teplotní Anomálie",
        linewidth=3,
        color="navy",
    )
    plt.plot(
        data["Year"],
        data["5-Year Mean"],
        label="5-letý průměr",
        color="red",
        linewidth=3,
    )
    plt.xlabel("Rok", fontsize=font_size)
    plt.ylabel("Teplotní Anomálie (°C)", fontsize=font_size)
    plt.tick_params(
        axis="both", which="major", labelsize=font_size - 5
    )  # Increase tick label font size
    plt.legend(fontsize=font_size)
    plt.grid()
    if save_file_path:
        plt.savefig(save_file_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main():
    dataset_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/tavg/land_ocean/1/4/1850-2025/data.csv"
    dataset_folder = "../../data"
    dataset_file = os.path.join(dataset_folder, "global_temperature.csv")
    save_plot_path = "../../data/plots/global_temperature_plot.png"

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    if not os.path.exists(save_plot_path):
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

    download_dataset(dataset_url, dataset_file)
    plot_global_temperature(dataset_file, save_plot_path)


if __name__ == "__main__":
    main()
