from concurrent.futures import ThreadPoolExecutor

def sum(a, b):
    return a + b

def operation(a, b, operation_function):
    return operation_function(a, b)

if __name__ == "__main__":
    my_a = [1, 2, 3, 4, 5]
    my_b = [10, 20, 30, 40, 50]

    print(f"Calculating sums of {my_a} and {my_b} using ThreadPoolExecutor...")

    with ThreadPoolExecutor() as executor:
        # You sould expand the constant parameter too, to the length of the other iterables
        my_sums = executor.map(operation, my_a, my_b, [sum]*len(my_a))

    print(f"Results: {list(my_sums)}")

