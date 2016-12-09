import timesynth as ts
import numpy as np
import ujson as json
import tqdm

#Function for building the dictionary
def create_data_dictionary(phi, sigma, irregular_time_samples, signal):
    data_dict = {}
    data_dict['phi'] = phi
    data_dict['sigma'] = sigma
    data_dict['time_samples'] = list(irregular_time_samples)
    data_dict['signal'] = list(signal)
    return data_dict


def write_file(folder_name, file_name, data_dict):
    with open(folder_name + file_name + '.json', 'w') as fp:
        json.dump(data_dict, fp)

def generate_data(phi, sigma, num_points):
    time_sampler = ts.TimeSampler(stop_time=20)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=num_points,
                                                                keep_percentage=50)
    irregular_time_samples_diff = np.diff(np.append(0, irregular_time_samples))
    signal = np.zeros(len(irregular_time_samples)+1)
    noise_samples = np.random.normal(size=len(irregular_time_samples))
    for i in range(len(irregular_time_samples)):
        signal[i+1] = np.power(phi, irregular_time_samples_diff[i])*signal[i] + \
                        sigma*np.sqrt(1 - np.power(phi, 2*irregular_time_samples_diff[i]))*noise_samples[i]
    return irregular_time_samples, signal[1:]


def main(num_samples):
    # Only run to generate datasets
    # Change folder_name argument for write_file function to generate the dataset
    phi_samples = np.random.uniform(size=num_samples)
    sigma_samples = np.random.uniform(size=num_samples)
    signal_lengths = np.random.randint(low=500, high=1000, size=num_samples)
    for i in tqdm.tqdm(range(num_samples)):
        sigma = sigma_samples[i]
        phi = phi_samples[i]
        sig_length = signal_lengths[i]
        signal = np.nan
        while np.any(np.isnan(signal)):
            time_samples, signal = generate_data(phi, sigma, 600)
        data_dict = create_data_dictionary(phi, sigma, time_samples, signal)
        file_name = 'series_'+str(i)
        write_file('../data/corr/train/', file_name, data_dict)

if __name__ == "__main__":
    main(100000)
