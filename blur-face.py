import cv2
import os

def blur_faces_single_image(image_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ไม่พบภาพที่ระบุ: {image_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"ในภาพ {os.path.basename(image_path)} พบใบหน้าทั้งหมด {len(faces)} ใบ")

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_blurred = cv2.GaussianBlur(face, (51, 51), 30)
        image[y:y+h, x:x+w] = face_blurred

    cv2.imwrite(output_path, image)
    print(f"บันทึกภาพแล้วที่: {output_path}")
    return True

def blur_faces_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"blurred_{filename}")

        blur_faces_single_image(input_path, output_path)

if __name__ == "__main__":
    input_folder = "images_input"   
    output_folder = "images_output" 
    blur_faces_in_folder(input_folder, output_folder)
