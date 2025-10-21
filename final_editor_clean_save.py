import cv2
import dlib
import numpy as np
import argparse
import os

def run_final_editor(image_path, predictor_path):
    """
    A fully automatic editor that rotates eyes (without eyebrows) and mouth,
    providing a real-time slider to adjust the blend area and saving the clean image.
    """
    # --- 1. ONE-TIME SETUP AND PRE-COMPUTATION ---
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return
        
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (h, w) = original_image.shape[:2]

    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    if not faces:
        print("Error: No face detected in the image.")
        return
        
    all_landmarks = np.array([(p.x, p.y) for p in predictor(gray, faces[0]).parts()])

    left_eye_pts = all_landmarks[36:42]
    right_eye_pts = all_landmarks[42:48]
    mouth_pts = all_landmarks[48:68]
    
    features = {
        "left_eye": {"points": left_eye_pts},
        "right_eye": {"points": right_eye_pts},
        "mouth": {"points": mouth_pts}
    }

    print("Pre-computing rotations...")
    for key, data in features.items():
        hull = cv2.convexHull(data["points"])
        M = cv2.moments(hull)
        if M["m00"] == 0: continue
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        data["center"] = center
        rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.0)
        data["rotated_image"] = cv2.warpAffine(original_image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        data["base_rect"] = cv2.boundingRect(hull)

    # --- 2. INTERACTIVE WINDOW AND REAL-TIME LOOP ---
    
    window_name = "Final Editor"
    cv2.namedWindow(window_name)
    
    def on_change(val): pass

    cv2.createTrackbar("Padding", window_name, 10, 50, on_change)

    while True:
        padding = cv2.getTrackbarPos("Padding", window_name)
        
        # This is the CLEAN canvas for processing. It gets recreated every frame.
        processed_image = original_image.copy()
        
        for key, data in features.items():
            if "rotated_image" not in data: continue
            
            x, y, w, h = data["base_rect"]
            ex = max(0, x - padding)
            ey = max(0, y - padding)
            ew = w + (padding * 2)
            eh = h + (padding * 2)
            
            blend_center = (ex + ew // 2, ey + eh // 2)
            
            source_roi = data["rotated_image"][ey:min(ey+eh, original_image.shape[0]), ex:min(ex+ew, original_image.shape[1])]
            if source_roi.size == 0: continue

            mask = 255 * np.ones(source_roi.shape, source_roi.dtype)
            
            # Blend onto the clean canvas
            processed_image = cv2.seamlessClone(source_roi, processed_image, mask, blend_center, cv2.NORMAL_CLONE)

        # --- CORRECTION IS HERE ---
        # 1. Create a separate copy for display purposes
        display_image = processed_image.copy()

        # 2. Draw the text ONLY on the display copy
        cv2.putText(display_image, f"Padding: {padding}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_image, "Press 's' to save, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 3. Show the image that has text on it
        cv2.imshow(window_name, display_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 4. Save the CLEAN image (processed_image), which never had text drawn on it
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_final_processed{ext}"
            cv2.imwrite(output_path, processed_image)
            print(f"✅ Image successfully saved to: {output_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final real-time editor with clean save.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    args = parser.parse_args()
    
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_model):
        print(f"Error: Predictor file not found at '{predictor_model}'")
    else:
        run_final_editor(args.image, predictor_model)