import subprocess
import sys

def generate_range(start, end, step, multiplier):
    data = []
    for value in range(int(start * multiplier), int(end * multiplier), int(step * multiplier)):
        value_to_append = int(value / multiplier) if float.is_integer(value / multiplier) else value / multiplier
        data.append(value_to_append)

    return data

image_dim_range = generate_range(20, 100, 10, 1)
train_split_range = generate_range(0.7, 0.9, 0.5, 10)
learning_rate_range = generate_range(0.0001, 0.01, 0.0005, 10000)
epochs_range = generate_range(30, 200, 10, 1)
epsilon_range = generate_range(1e-08, 1e-06, 5e-07, 1e08)
kernel_size_range = generate_range(2, 5, 1, 1)
dense_neurons_range = generate_range(50, 200, 10, 1)

if __name__ == '__main__':
    for img_dim in image_dim_range:
        for train_split in train_split_range:
            for learning_rate in learning_rate_range:
                for epochs in epochs_range:
                    for epsilon in epsilon_range:
                        for kernel_size in kernel_size_range:
                            for dense_neurons in dense_neurons_range:

                                print(subprocess.run([
                                    f'python3 main.py -i {img_dim} -lr {learning_rate} -e {epochs} -l {epsilon} -k {kernel_size} -d {dense_neurons} -s {train_split}'],
                                    shell=True, stderr=sys.stderr, stdout=sys.stdout
                                ))

