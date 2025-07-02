import matplotlib.pyplot as plt
from PIL import Image

def show_image_with_label(image_path, label):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()

def plot_video_timeline(frame_results):
    times = [f['timestamp'] for f in frame_results]
    labels = [f['label'] for f in frame_results]
    plt.figure(figsize=(12, 2))
    plt.scatter(times, [1]*len(times), c='b')
    for t, l in zip(times, labels):
        plt.text(t, 1.02, l, rotation=45, ha='right', va='bottom', fontsize=8)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title("Video Frame Predictions Timeline")
    plt.tight_layout()
    plt.show()
