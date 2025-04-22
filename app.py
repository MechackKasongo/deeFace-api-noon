from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
import cv2
import tempfile

app = Flask(__name__)
CORS(app)

root_dir = "dataset"

def check_image(path):
    img = cv2.imread(path)
    return img is not None

@app.route('/recognize', methods=['POST'])
def recognize_person():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    uploaded_file = request.files['image']

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        uploaded_file.save(tmp.name)
        input_image_path = tmp.name

    if not check_image(input_image_path):
        return jsonify({"error": "Image invalide"}), 400

    members = {}

    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if os.path.isdir(person_dir):
            member_images = []
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                if check_image(image_path):
                    member_images.append(image_path)
            if member_images:
                members[person_name] = member_images

    for member, images in members.items():
        for image in images:
            try:
                result = DeepFace.verify(input_image_path, image, model_name='VGG-Face')
                if result['verified']:
                    confidence = result['distance']
                    return jsonify({
                        "person": member,
                        "confidence": f"{100 * (1 - confidence):.2f}%",
                        "reference_image": image
                    })
            except Exception as e:
                print(f"Erreur avec {member} : {e}")

    return jsonify({"message": "Aucune correspondance trouvée"}), 404

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)



#####################################################################################

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from deepface import DeepFace
# import os
# import cv2

# app = Flask(__name__)
# CORS(app)

# # Utiliser un dossier local au projet (sur Render)
# root_dir = "dataset"

# def check_image(path):
#     img = cv2.imread(path)
#     return img is not None

# @app.route('/recognize', methods=['POST'])
# def recognize_person():
#     data = request.get_json()
#     input_image_path = data.get('image_path')

#     if not input_image_path or not check_image(input_image_path):
#         return jsonify({"error": "Image invalide ou introuvable"}), 400

#     members = {}

#     for person_name in os.listdir(root_dir):
#         person_dir = os.path.join(root_dir, person_name)
#         if os.path.isdir(person_dir):
#             member_images = []
#             for image_file in os.listdir(person_dir):
#                 image_path = os.path.join(person_dir, image_file)
#                 if check_image(image_path):
#                     member_images.append(image_path)
#             if member_images:
#                 members[person_name] = member_images

#     for member, images in members.items():
#         for image in images:
#             try:
#                 result = DeepFace.verify(input_image_path, image, model_name='VGG-Face')
#                 if result['verified']:
#                     confidence = result['distance']
#                     return jsonify({
#                         "person": member,
#                         "confidence": f"{100 * (1 - confidence):.2f}%",
#                         "reference_image": image
#                     })
#             except Exception as e:
#                 print(f"Erreur avec {member} : {e}")

#     return jsonify({"message": "Aucune correspondance trouvée"}), 404

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)
