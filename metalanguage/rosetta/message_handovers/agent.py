class Agent:
    def __init__(self, name):
        self.name = name
        self.counter = 0

def send_message(sender, receiver):
    receiver.counter += 1
    print(f"{sender.name} -> {receiver.name}: {receiver.counter}")

if __name__ == "__main__":
    alice = Agent("Alice")
    bob = Agent("Bob")
    send_message(alice, bob)
    send_message(alice, bob)
