import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("hello_world")

    from modules.hello_world import hello_world
    print(hello_world())