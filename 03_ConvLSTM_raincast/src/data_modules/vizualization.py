from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import torch

def visualize_frame(tensor):
    # Convert tensor to numpy array from PyTorch tensor
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()

    # Plot the tensor using matplotlib with blue to green to red color map
    plt.imshow(tensor, cmap='seismic')
    plt.colorbar()
    plt.title("2D Tensor Visualization")
    plt.show()

def visualize_batch_tensor_interactive(tensor, batch_idx=0, name=None):
    """
    Creates an interactive visualization of a 5D tensor with shape [batch, sequence, channels, height, width]
    Parameters:
        tensor: PyTorch tensor of shape [batch, sequence, channels, height, width]
        batch_idx: Which sample from the batch to visualize
        name: Optional name for the visualization
    """
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(tensor):
        # Select specific sample from batch and squeeze the channel dimension
        tensor = tensor[batch_idx].squeeze(1).numpy()  # Now shape is [sequence, height, width]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Display initial frame
    frame_idx = 0
    # For tensor [sequence, height, width], we display frame_idx slice
    img = ax.imshow(tensor[frame_idx], cmap='viridis')
    plt.colorbar(img)

    # Create slider axis and slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Sequence',
        valmin=0,
        valmax=tensor.shape[0] - 1,  # Number of frames in sequence
        valinit=frame_idx,
        valstep=1
    )

    # Set title
    if name:
        title = ax.set_title(f'{name} - Frame {frame_idx} / {tensor.shape[0]-1} (Batch {batch_idx})')
    else:
        title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[0]-1} (Batch {batch_idx})')

    def update(val):
        frame = int(slider.val)
        img.set_array(tensor[frame])
        if name:
            title.set_text(f'{name} - Frame {frame} / {tensor.shape[0]-1} (Batch {batch_idx})')
        else:
            title.set_text(f'Frame {frame} / {tensor.shape[0]-1} (Batch {batch_idx})')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()



def visualize_tensor_interactive(tensor,name):
    """
    Creates an interactive visualization of a 3D tensor with shape [height, width, frames]
    Parameters:
        tensor: PyTorch tensor or numpy array of shape [height, width, frames]
    """
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Display initial frame
    frame_idx = 0
    img = ax.imshow(tensor[:, :, frame_idx], cmap='viridis')
    plt.colorbar(img)

    # Create slider axis and slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=tensor.shape[2] - 1,
        valinit=frame_idx,
        valstep=1
    )

    # Title with frame information
    title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[2]-1}')

    # Update function for the slider
    if name:
        title = ax.set_title(f'{name} - Frame {frame_idx} / {tensor.shape[2]-1}')
    else:
        title = ax.set_title(f'Frame {frame_idx} / {tensor.shape[2]-1}')

        # Modified update function to maintain the name in the title
    def update(val):
        frame = int(slider.val)
        img.set_array(tensor[:, :, frame])
        if name:
            title.set_text(f'{name} - Frame {frame} / {tensor.shape[2]-1}')
        else:
            title.set_text(f'Frame {frame} / {tensor.shape[2]-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def visualize_random_sample(file_name):
    sevirDataSet = SEVIRDataset(file_name)

    random = torch.randint(0, 552, (1,)).item()
    sample0 = SEVIRDataset.__getitem__(random)
    visualize_tensor_interactive(sample0,f"Random sample(id:{random}) from: {file_name}")
