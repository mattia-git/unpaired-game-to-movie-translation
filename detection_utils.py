import cv2
import numpy as np

# Feature matching
detector = cv2.SIFT_create()
matcher = cv2.FlannBasedMatcher()

def patches_are_different(patch1, patch2, similarity_threshold=0.2):
    """
    Compare two patches to determine if they are different
    :param patch1: First patch
    :param patch2: Second patch
    :param similarity_threshold: Similarity threshold
    """
    # Patches are tensors, convert to grayscale image first
    patch1 = patch1.squeeze().cpu().numpy().transpose(1, 2, 0)
    patch2 = patch2.squeeze().cpu().numpy().transpose(1, 2, 0)

    # # Convert to grayscale
    patch1 = cv2.cvtColor(patch1, cv2.COLOR_RGB2GRAY)
    patch2 = cv2.cvtColor(patch2, cv2.COLOR_RGB2GRAY)

    # to 8bit   
    patch1 = cv2.normalize(patch1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    patch2 = cv2.normalize(patch2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(patch1, None)
    kp2, des2 = detector.detectAndCompute(patch2, None)

    # Match descriptors
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate similarity score
    similarity_score = len(good_matches) / max(len(kp1), len(kp2))

    return similarity_score < similarity_threshold

def is_illuminated(frame, threshold=0.09):
    """
    Check if a frame is illuminated
    :param frame: Frame to check
    :param threshold: Illumination threshold
    """
    _frame = frame.detach().cpu().numpy().transpose(1, 2, 0)
    _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2GRAY)

    # Calculate the mean pixel intensity
    mean_intensity = np.mean(_frame)

    # Return whether the frame is illuminated
    return mean_intensity > threshold

def is_large(frame, threshold=0.15, max_area=409600):
    """
    Check if a frame is large
    :param frame: Frame to check
    :param threshold: Size threshold
    :param max_area: Maximum area that a frame can have
    """
    width, height = frame.shape[1], frame.shape[2]
    area = width * height
    return area > max_area * threshold

def is_blurry(frame, threshold=0.1):
    """
    Detect blurriness in a frame using FFT
    :param frame: Frame to check
    :param threshold: Blurriness threshold
    """
    frame = frame.detach().cpu().numpy().transpose(1, 2, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Calculate the FFT
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Calculate the mean pixel intensity
    mean_intensity = np.mean(magnitude_spectrum)

    # Return whether the frame is blurred
    return mean_intensity < threshold

def accept_patch(patch, lookback):
    """
    Check if a patch should be accepted
    :param patch: Patch to check
    :param lookback: Lookback buffer
    """

    if not is_illuminated(patch):
        return False
    
    if not is_large(patch):
        return False
    
    if is_blurry(patch):
        return False
    
    for prev_patch in lookback:
        if not patches_are_different(patch, prev_patch):
            return False
        
    return True

def add_and_pop(lookback, patch, max_length=10):
    """
    Add a patch to the lookback buffer and pop the oldest patch
    :param lookback: Lookback buffer
    :param patch: Patch to add
    :param max_length: Maximum length of the lookback buffer
    """
    lookback.append(patch)

    if len(lookback) > max_length:
        lookback.pop(0)