import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

# flags
READ_FROM_CSV = False
PLOT_SHOW = False
PLOT_SAVE = False
SAVE_TO_CSV = not READ_FROM_CSV
RATIO = 10
dataset_dir = "sinusoidal_dataset"

if(PLOT_SAVE or SAVE_TO_CSV):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

# Number of stations
NUM_OF_STATIONS = 6
NUM_OF_SAMPLES_PER_STATION = 100

def get_stations():
    # station identifiers are hard-coded 6 coordinate values (latitude, longitude), between latitude 36.54 and 42.93, and longitude 27.92 and 42.08,
    # these intervals are the approximate latitude and longitude of Turkey start and end points
    latitudes = np.linspace(36.54, 42.93, 2)
    longitudes = np.linspace(27.92, 42.08, 3)

    # Create combinations of latitudes and longitudes
    grid_points = [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]

    # Select 6 stations from the grid (ensuring even distribution if possible)
    stations = grid_points[:NUM_OF_STATIONS]

    if PLOT_SHOW or PLOT_SAVE:
        # Plot the stations on a 2D grid
        plt.figure(figsize=(8, 6))
        for i, (lat, lon) in enumerate(stations):
            plt.scatter(lon, lat, label=f"Station {i+1}", s=100)

        # Annotate the stations
        for i, (lat, lon) in enumerate(stations):
            plt.text(lon + 0.2, lat, f"{i+1}", fontsize=9)

        # Grid details
        plt.title("Stations in 2D Grid")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        if(PLOT_SAVE):
            plt.savefig(os.path.join(dataset_dir, "stations.png"))
        if(PLOT_SHOW):
            plt.show()
    return stations

def generate_sinusoidal_mixture(frequency_range, n_samples, sampling_rate):
    t = np.linspace(0, duration, n_samples, endpoint=False)  # Time vector
    base_phase = np.random.uniform(0, 2 * np.pi)  # Shared phase for all channels in a sample
    n_waves = np.random.randint(10, 16)  # Random number of waves to mix (10-15 waves)
    base_frequencies = [np.random.uniform(*frequency_range) for _ in range(n_waves)]  # Shared frequencies
    base_amplitudes = [np.random.uniform(0.7, 1.3) for _ in range(n_waves)]  # Shared amplitudes
    mixtures = []

    for _ in range(3):
        mixture = np.zeros(n_samples)
        channel_factor = np.random.uniform(0.5, 1.5)  # Random amplitude scaling for channels
        for freq, amp in zip(base_frequencies, base_amplitudes):
            amplitude = amp * channel_factor
            mixture += amplitude * np.sin(2 * np.pi * freq * t + base_phase)
        mixtures.append(mixture)

    return mixtures

stations = get_stations()

# since most of the earthquakes have frequencies less than 20 Hz, we can set the frequency ranges as follows:
freq_starts = np.linspace(1, 20, NUM_OF_STATIONS)
freq_ends = np.linspace(4, 24, NUM_OF_STATIONS)

# Constants
sampling_rate = 1000  # 1000 samples per second
duration = 60  # seconds
n_samples = sampling_rate * duration  # Total number of samples per channel

frequency_ranges = {
    i: (int(freq_starts[i]), int(freq_ends[i])) for i in range(NUM_OF_STATIONS)
}  # Frequency ranges per station

# if printed: {0: (1, 4), 1: (4, 8), 2: (8, 12), 3: (12, 16), 4: (16, 20), 5: (20, 24)}

if not READ_FROM_CSV:
    # Generate dataset
    data = []
    for i in range(len(stations)):
        for _ in range(NUM_OF_SAMPLES_PER_STATION):
            channels = generate_sinusoidal_mixture(frequency_ranges[i], n_samples, sampling_rate)
            
            # Combine station and channels into a single dictionary
            sample = {
                'Station': tuple(stations[i]),
                'Channel_1': json.dumps(channels[0].tolist()),
                'Channel_2': json.dumps(channels[1].tolist()),
                'Channel_3': json.dumps(channels[2].tolist())
            }
            data.append(sample)

    # Convert dataset to DataFrame
    data_df = pd.DataFrame(data)

    if SAVE_TO_CSV:
        # Save to file
        output_file = "sinusoidal_dataset.csv"
        data_df.to_csv(os.path.join(dataset_dir, output_file), index=False)
        print(f"Dataset saved to {os.path.join(dataset_dir, output_file)}")    
else:  # read from csv
    csv_path = os.path.join(dataset_dir, "sinusoidal_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Set READ_FROM_CSV = False to generate the dataset.")
    # Read the CSV file
    # Read the CSV file
    data_df = pd.read_csv(csv_path)
    print("data head: ", data_df.head())
    data_df['Station'] = data_df['Station'].apply(lambda x: eval(x))

if PLOT_SHOW or PLOT_SAVE:
    # Plot samples
    fig, axes = plt.subplots(6, 2, figsize=(15, 10))
    axes = axes.flatten()
    time_vector = np.linspace(0, duration, n_samples, endpoint=False)
    ratio = RATIO
    short_time_vector = time_vector[:n_samples // ratio]  # Use only 1/ratio of the sample
 
    for st in range(len(stations)):
        # Filter samples for the current station
        samples = data_df[data_df['Station'] == stations[st]].iloc[:2]
        print(f"Station {st+1}: {len(samples)} samples")
        for i, (_, sample) in enumerate(samples.iterrows()): 
            ax = axes[st * 2 + i]
            ax.plot(short_time_vector, json.loads(sample['Channel_1'])[:n_samples // ratio], label='Channel 1')
            ax.plot(short_time_vector, json.loads(sample['Channel_2'])[:n_samples // ratio], label='Channel 2')
            ax.plot(short_time_vector, json.loads(sample['Channel_3'])[:n_samples // ratio], label='Channel 3')
            ax.set_title(f"Station {st}, Sample {i + 1}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Amplitude", fontsize=9)
            ax.grid(True)
            ax.legend(fontsize=8, loc='upper right')

    # Adjust layout
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    if(PLOT_SAVE):
        plt.savefig(os.path.join(dataset_dir, "example_samples_60sec_w_ratio_"+str(ratio)+".png"))
    if(PLOT_SHOW):
        plt.show()