from PIL import Image
import os

def gif_convert(path=r"C:\Users\lenovo\Projects\Neural_Style_Transfer\output\style_transfer", explicit_image=r"final_stylized.jpg" , starts_with = "iter"):
    # Filter for images that start with "iter" and have valid extensions
    iter_images = [
        f for f in os.listdir(path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        and f.startswith(starts_with)
    ]
    
    # Sort the images that start with "iter"
    iter_images = sorted(iter_images)
    
    # Build the final list of images: start with iter_images
    final_images = iter_images.copy()
    
    # If an explicit image is provided, try to add it to the end
    if explicit_image:
        explicit_image_path = os.path.join(path, explicit_image)
        if os.path.exists(explicit_image_path):
            # Remove the image if it's already in the list, then append it
            if explicit_image in final_images:
                final_images.remove(explicit_image)
            final_images.append(explicit_image)
        else:
            print(f"Explicit image '{explicit_image}' not found in the folder.")
    
    # Open images and store them in a list
    images = [Image.open(os.path.join(path, file)) for file in final_images]

    # Check that we have at least one image
    if images:
        # Save as an animated GIF
        images[0].save("output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
        print("GIF created successfully!")
    else:
        print("No images found to create a GIF.")

# Call the function with an explicit image, or use None if not needed
# gif_convert(explicit_image="content_final.jpg")
# To run without an explicit image, you can also call:

# gif_convert(explicit_image=None , path = r"C:\Users\lenovo\Projects\Neural_Style_Transfer\feature_maps" , starts_with="Layer")
# gif_convert(explicit_image=None)

# gif_convert(explicit_image="reconstruction.jpg" ,path=r"C:\Users\lenovo\Projects\Neural_Style_Transfer\output\content" ,starts_with="iter")
gif_convert(explicit_image=None ,path=r"C:\Users\lenovo\Projects\Neural_Style_Transfer\output\style" ,starts_with="iter")
