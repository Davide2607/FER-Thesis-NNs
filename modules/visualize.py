import matplotlib.pyplot as plt

def plot_image(image, title="No Title"):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()