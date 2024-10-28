import pickle
import os

dir_path = "transpiled_circuit"

print(os.listdir(dir_path))

for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    print(path)
    with open(path, mode='rb') as f:
        m = pickle.load(f)[0]
    with open(f'read_file/{filename}', mode='w') as f:
        rows = m.shape[0]
        cols = m.shape[1]
        f.write(f'{rows} {cols}\n')
        for col in range(cols):
            for row in range(rows):
                if m[row, col] == 0 + 0j:
                    continue
                if m[row, col].real < 10:
                    f.write(f'{m[row, col].real + 10 + m[row, col].imag} {row} {col}\n')
                else:
                    f.write(f'{20} {row} {col}\n')