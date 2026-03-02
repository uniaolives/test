from eternity_crystal import EternityCrystal

class MockLargeData:
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size

def test_eternity_crystal():
    crystal = EternityCrystal()
    print(f"Crystal initialized with capacity {crystal.genome_capacity_tb}TB")

    # Human genome @150x is 450GB
    size = int(450 * 1e9)
    genome_data = MockLargeData(size)

    is_valid = crystal.validate_archive(genome_data)
    print(f"Validation result: {is_valid}")

    assert is_valid == True

    # Test invalid size
    is_valid_wrong_size = crystal.validate_archive(b"too small")
    print(f"Validation result (wrong size): {is_valid_wrong_size}")
    assert is_valid_wrong_size == False

if __name__ == "__main__":
    test_eternity_crystal()
    print("Python test passed!")
