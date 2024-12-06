import matplotlib.pyplot as plt

def read_loss_file(file_name):
    """Reads a loss file and returns two lists: x values and y values."""
    x_vals = []
    y_vals = []
    with open(file_name, 'r') as file:
        for line in file:
            try:
                x, y = line.split(':')
                x_vals.append(int(x.strip()))
                y_vals.append(float(y.strip()))
            except ValueError:
                # If a line cannot be processed, print a message and skip it
                print(f"Skipping invalid line: {line.strip()}")
    return x_vals, y_vals

def plot_losses(eval_loss_file, train_loss_file):
    # Read the loss data from both files
    eval_x, eval_y = read_loss_file(eval_loss_file)
    train_x, train_y = read_loss_file(train_loss_file)


    train_x = train_x[:len(eval_x)]
    train_y = train_y[:len(eval_y)]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(eval_x, eval_y, label='Evaluation Loss', color='blue', linewidth=2)
    plt.plot(train_x, train_y, label='Training Loss', color='red', linewidth=2)

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

# Replace with the actual paths to your loss files
origin = "./results/mlm_EmbBERT/"
eval_loss_file = origin + 'eval_loss.txt'
train_loss_file = origin + 'train_loss.txt'

# Plot the loss values
plot_losses(eval_loss_file, train_loss_file)
