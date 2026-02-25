import cv2
import imageio


def create_video(file_path_images: list[str], file_path_video: str, fps: int = 5, **kwargs) -> None:
    """Create a video from a list of images.

    Args:
        file_path_images (list[str]): List of file paths to images.
        file_path_video (str): File path for the output video.
        fps (int, optional): Frames per second for the video. Defaults to 5.
        **kwargs: Additional keyword arguments to pass to :func:`cv2.VideoWriter`.
    """
    # Get frame size
    image = cv2.imread(file_path_images[0])
    frame_size = (image.shape[1], image.shape[0])

    # Initialize the video writer with codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    out = cv2.VideoWriter(file_path_video, fourcc, fps, frame_size, **kwargs)

    # Write each image to the video
    for file_path_image in file_path_images:
        image = cv2.imread(file_path_image)
        image = cv2.resize(image, frame_size)
        out.write(image)

    # Release the video writer
    out.release()


def create_gif(file_path_images: list[str], file_path_gif: str, fps: int = 5, **kwargs) -> None:
    """Create a GIF from a list of images.

    Args:
        file_path_images (list[str]): List of file paths to images.
        file_path_gif (str): File path for the output GIF.
        fps (int, optional): Frames per second for the GIF. Defaults to 5.
        **kwargs: Additional keyword arguments to pass to :func:`imageio.mimsave`.
    """
    # Set default loop to 0 (infinite) if not provided
    kwargs.setdefault("loop", 0)

    # Read images
    images = [imageio.imread(file_path_image) for file_path_image in file_path_images]

    # Save as GIF
    imageio.mimsave(file_path_gif, images, fps=fps, **kwargs)
