import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import os

def feature_visualization(feature_maps, layer_name="Layer 19"):
    """
    Visualizes first 16 channels of feature maps from a single layer
    """
    if feature_maps is None:
        raise ValueError("No feature maps detected!")
    
    features = feature_maps.detach().cpu().squeeze(0)
    num_channels = min(16, features.shape[0])  # Show first 16 channels
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Style Feature Maps: {layer_name}", fontsize=14)
    
    for i in range(num_channels):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i+1}")
    
    plt.tight_layout()
    plt.show()

def heat_map(gram_matrix , layer_name = ""):
    """
    Visulaize the Correlations between the feature maps of the layer
    """
    gram_np = gram_matrix.squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(gram_np, interpolation='nearest', cmap='gray')
    plt.title("Gram Matrix")
    plt.colorbar()
    plt.show()

def plot_loss_history(total_loss_history, style_loss_history=None, content_loss_history=None, tv_loss_history=None):
    """Plot the total loss and any provided loss components over time."""
    plt.figure(figsize=(12, 5))
    
    # Plot Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(total_loss_history, label='Total Loss')
    plt.title('Total Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot available loss components
    plt.subplot(1, 2, 2)
    if style_loss_history is not None:
        plt.plot(style_loss_history, label='Style Loss')
    if content_loss_history is not None:
        plt.plot(content_loss_history, label='Content Loss')
    if tv_loss_history is not None:
        plt.plot(tv_loss_history, label='TV Loss')
    
    plt.title('Loss Components')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def show_progress(img_tensor, step, total_steps, final=False):
        """Display progress of style reconstruction."""
        # Convert tensor to image
        img = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        if final:
            plt.title(f"Final Result")
        else:
            plt.title(f"Step {step}/{total_steps}")
        plt.axis('off')
        plt.show()

def plot_loss(
    total_loss_history,
    content_loss_history=None,
    style_loss_history=None,
    tv_loss_history=None,
    output_path="loss_graph"
):
    """Plot and save available loss values as graph and CSV."""
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot Total Loss
    if total_loss_history is not None and len(total_loss_history) > 0:
        plt.subplot(2, 1, 1)
        plt.plot(total_loss_history, 'b-', label='Total Loss')
        plt.title("Total Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
    else:
        print("Warning: total_loss_history is empty or None. Skipping plot.")

    # Plot individual components
    plt.subplot(2, 1, 2)
    plotted = False
    if content_loss_history is not None and len(content_loss_history) > 0:
        plt.plot(content_loss_history, 'g-', label='Content Loss')
        plotted = True
    if style_loss_history is not None and len(style_loss_history) > 0:
        plt.plot(style_loss_history, 'm-', label='Style Loss')
        plotted = True
    if tv_loss_history is not None and len(tv_loss_history) > 0:
        plt.plot(tv_loss_history, 'r-', label='TV Loss')
        plotted = True

    if plotted:
        plt.title("Loss Components")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No component losses to plot', horizontalalignment='center', verticalalignment='center')
        print("Warning: No component loss histories provided.")

    plt.tight_layout()
    save_path = os.path.join(output_path, "loss_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

    # Save as CSV (only for available losses)
    # iterations = np.arange(1, len(total_loss_history) + 1) if total_loss_history else []
    # losses = [iterations]

    # headers = ["iteration"]
    # if content_loss_history is not None and len(content_loss_history) > 0:
    #     losses.append(content_loss_history)
    #     headers.append("content_loss")
    # if style_loss_history is not None and len(style_loss_history) > 0:
    #     losses.append(style_loss_history)
    #     headers.append("style_loss")
    # if tv_loss_history is not None and len(tv_loss_history) > 0:
    #     losses.append(tv_loss_history)
    #     headers.append("tv_loss")
    # if total_loss_history is not None and len(total_loss_history) > 0:
    #     losses.append(total_loss_history)
    #     headers.append("total_loss")

    # if len(losses) > 1:
    #     stacked = np.column_stack(losses)
    #     np.savetxt(
    #         os.path.join(output_path, "loss_values.csv"),
    #         stacked,
    #         delimiter=",",
    #         header=",".join(headers),
    #         comments=""
    #     )
    #     print("Loss values saved as CSV.")
    # else:
    #     print("No losses to save as CSV.")