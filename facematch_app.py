import streamlit as st
import os
import json
import pickle
import face_recognition
from PIL import Image
import numpy as np
import cv2
import datetime

# Define dataset paths
DATASET_FOLDER = "dataset"
DETAILS_FILE = "details.json"
MODEL_FILE = "trained_model.pkl"

# Ensure dataset directory exists
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

# Load user details
if os.path.exists(DETAILS_FILE):
    with open(DETAILS_FILE, "r") as f:
        user_details = json.load(f)
else:
    user_details = {}

# Function to calculate age
def calculate_age(birthdate):
    birth_date = datetime.datetime.strptime(birthdate, "%Y-%m-%d")
    today = datetime.datetime.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

# Load dataset
def load_dataset():
    known_encodings, known_names, known_images = [], [], []
    for person in os.listdir(DATASET_FOLDER):
        person_path = os.path.join(DATASET_FOLDER, person)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                image_path = os.path.join(person_path, file)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_names.append(person)
                    known_images.append(image_path)
    return known_encodings, known_names, known_images

# Train and save model
def train_and_save_model():
    known_encodings, known_names, known_images = load_dataset()
    data = {"encodings": known_encodings, "names": known_names, "images": known_images}
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(data, file)
    return len(known_encodings)

# Categorize matches by similarity
def find_match(image):
    unknown_encoding = face_recognition.face_encodings(image)
    if not unknown_encoding:
        return None
    
    with open(MODEL_FILE, "rb") as file:
        data = pickle.load(file)
    
    known_encodings, known_names, known_images = data["encodings"], data["names"], data["images"]
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])
    
    matched_results = {}
    for i, match in enumerate(matches):
        if match:
            name = known_names[i]
            if name not in matched_results:
                matched_results[name] = []
            matched_results[name].append((known_images[i], face_distances[i]))
    
    # Sort images by lowest face distance (better match) & earliest timestamp
    for name in matched_results:
        matched_results[name].sort(key=lambda x: (x[1], os.path.getctime(x[0])))
    
    return matched_results

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Dataset", "Search Faces", "Webcam Search"])

if page == "Train Dataset":
    st.title("Train Face Recognition Model")
    name = st.text_input("Enter Name")
    birthdate = st.date_input("Enter Birthdate")
    birthplace = st.text_input("Enter Birthplace")
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files and name and birthdate and birthplace:
        person_folder = os.path.join(DATASET_FOLDER, name.strip())
        os.makedirs(person_folder, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(person_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            user_details[name] = {"name": name, "birthdate": str(birthdate), "birthplace": birthplace}
        with open(DETAILS_FILE, "w") as f:
            json.dump(user_details, f, indent=4)
        st.success("Images uploaded successfully!")

    if st.button("Train Model"):
        num_faces = train_and_save_model()
        st.success(f"Model trained successfully with {num_faces} faces!") if num_faces > 0 else st.error("No faces found in dataset.")

elif page in ["Search Faces", "Webcam Search"]:
    st.title("FaceMatch AI")
    image_rgb = None

    if page == "Search Faces":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        photo = st.camera_input("Take a picture")
        if photo:
            image = Image.open(photo)
            image_rgb = np.array(image.convert("RGB"))

    if image_rgb is not None:
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        matched_results = find_match(image_rgb)

        if matched_results:
            st.subheader("🔍 Matched Faces")
            for name, images in matched_results.items():
                img_count = len(images)
                best_img_path, best_distance = images[0]
                user_info = user_details.get(name, {})
                birthdate = user_info.get("birthdate", "N/A")
                age = calculate_age(birthdate) if birthdate != "N/A" else "Unknown"
                birthplace = user_info.get("birthplace", "N/A")

                # Calculate match percentage (lower distance = higher similarity)
                match_percentage = round((1 - best_distance) * 100, 2)

                # Create a clickable card-like design for each matched person
                with st.container():
                    cols = st.columns([1, 4])  # Adjust column widths as needed
                    with cols[0]:
                        st.image(best_img_path, width=80)
                    with cols[1]:
                        if st.button(f"View Details: {name}"):
                            # Store matched images in session state and navigate to details page
                            st.session_state["selected_name"] = name
                            st.session_state["matched_images"] = images
                            st.rerun() # st.experimental_rerun()

                        st.markdown(f"### {name}")
                        st.markdown(f"**📅 Birthdate:** {birthdate} ({age} years old)")
                        st.markdown(f"**📍 Birthplace:** {birthplace}")
                        st.markdown(f"**🖼️ Matches:** {img_count} images")
                        st.markdown(f"**✅ Match Percentage:** {match_percentage}%")

                # Add a horizontal divider for better separation
                st.markdown("---")

        # Display matched images on a new page
        if "selected_name" in st.session_state and "matched_images" in st.session_state:
            st.title(f"Matched Images for {st.session_state['selected_name']}")
            matched_images = st.session_state["matched_images"]
            img_cols = st.columns(5)  # Display images in a grid with 5 columns
            for i, (img_path, distance) in enumerate(matched_images):
                match_percentage = round((1 - distance) * 100, 2)
                with img_cols[i % 5]:
                    st.image(img_path, width=100)
                    st.caption(f"Match: {match_percentage}%")
        else:
            st.error("❌ No match found. Try again!")
