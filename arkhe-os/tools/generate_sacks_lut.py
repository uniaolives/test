# arkhe-os/tools/generate_sacks_lut.py
import numpy as np

def generate_sacks_lut(max_entries: int = 1024, output_file: str = "sacks_lut.hex"):
    """
    Generate BRAM initialization file for Sacks spiral navigation.
    Entries: 64-bit word {prime[32], theta_fixed[16], neighbors_packed[16]}
    """
    # Mocking prime generation for the tool script
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] + [0]*1014

    entries = []
    for i in range(max_entries):
        p = primes[i]
        r = np.sqrt(i + 1)
        theta = (2 * np.pi * r) % (2 * np.pi)
        theta_fixed = int((theta / (2 * np.pi)) * 65535)
        neighbor_packed = 0x1234 # Mocked neighbors

        word = (p << 32) | (theta_fixed << 16) | neighbor_packed
        entries.append(f"{word:016X}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(entries))
    print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_sacks_lut()
