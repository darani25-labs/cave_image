import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to brighten or darken an image.
    
    Parameters:
    - image: Input image (numpy array)
    - gamma: Gamma value (>1 brightens, <1 darkens, =1 no change)
    
    Returns:
    - Gamma corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def brighten_cave_image(image_path, gamma_values=[1.5, 2.0, 2.5], output_path='output/cave_brightened.jpg'):
    """
    Brighten a dark cave image using gamma correction.
    
    Parameters:
    - image_path: Path to the dark cave image
    - gamma_values: List of gamma values to try (default: [1.5, 2.0, 2.5])
    - output_path: Path to save the best enhanced image
    """
    
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply different gamma corrections
    corrected_images = []
    for gamma in gamma_values:
        corrected = apply_gamma_correction(img, gamma)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        corrected_images.append((gamma, corrected, corrected_rgb))
    
    # Save the middle gamma value result (usually best balance)
    best_gamma_idx = len(gamma_values) // 2
    cv2.imwrite(output_path, corrected_images[best_gamma_idx][1])
    print(f"Enhanced image saved to: {output_path}")
    print(f"Used gamma value: {corrected_images[best_gamma_idx][0]}")
    
    # Display results
    num_images = len(gamma_values) + 1
    plt.figure(figsize=(18, 6))
    
    # Original image
    plt.subplot(1, num_images, 1)
    plt.imshow(img_rgb)
    plt.title('Original Dark Cave Image')
    plt.axis('off')
    
    # Corrected images
    for idx, (gamma, corrected_bgr, corrected_rgb) in enumerate(corrected_images, start=2):
        plt.subplot(1, num_images, idx)
        plt.imshow(corrected_rgb)
        plt.title(f'Gamma = {gamma}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display histogram comparison for the best result
    plt.figure(figsize=(15, 5))
    
    # Original histogram
    plt.subplot(1, 3, 1)
    plt.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Corrected histogram
    best_corrected = corrected_images[best_gamma_idx][1]
    plt.subplot(1, 3, 2)
    plt.hist(cv2.cvtColor(best_corrected, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], color='green', alpha=0.7)
    plt.title(f'Enhanced Image Histogram (γ={corrected_images[best_gamma_idx][0]})')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Side-by-side comparison
    plt.subplot(1, 3, 3)
    comparison = np.hstack([img_rgb, corrected_images[best_gamma_idx][2]])
    plt.imshow(comparison)
    plt.title('Before vs After')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corrected_images[best_gamma_idx][1]


def custom_gamma_correction(image_path, custom_gamma, output_path='output/cave_custom.jpg'):
    """
    Apply custom gamma correction value.
    
    Parameters:
    - image_path: Path to the cave image
    - custom_gamma: Your chosen gamma value
    - output_path: Path to save the result
    """
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Apply gamma correction
    corrected = apply_gamma_correction(img, custom_gamma)
    
    # Save result
    cv2.imwrite(output_path, corrected)
    print(f"Custom gamma corrected image saved to: {output_path}")
    print(f"Gamma value used: {custom_gamma}")
    
    # Display comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title(f'Gamma Corrected (γ={custom_gamma})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corrected


# Example usage
if __name__ == "__main__":
    # Replace with your cave image path
    image_path = "images/cave.jpg"
    
    print("Brightening dark cave image using gamma correction...")
    print("=" * 60)
    
    # Method 1: Try multiple gamma values
    enhanced_image = brighten_cave_image(
        image_path, 
        gamma_values=[1.5, 2.0, 2.5],
        output_path='output/cave_brightened.jpg'
    )
    
    # Method 2: Use custom gamma value (uncomment to use)
    # custom_gamma_correction(image_path, custom_gamma=2.2, output_path='output/cave_custom.jpg')
    
    print("\n" + "=" * 60)
    print("Gamma correction applied successfully!")
    print("The enhanced image shows improved visibility of cave details.")
    print("\nTip: Gamma values explained:")
    print("  - γ = 1.0: No change")
    print("  - γ = 1.5-2.0: Moderate brightening (good for slightly dark images)")
    print("  - γ = 2.5-3.0: Strong brightening (good for very dark images)")
    print("  - γ < 1.0: Darkening (rarely needed for cave images)")