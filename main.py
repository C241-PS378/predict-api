import os
import logging
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, Response, File, UploadFile
from PIL import Image
from google.cloud import storage
from datetime import datetime
from tensorflow.keras.utils import img_to_array
from keras.applications.mobilenet import preprocess_input

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sampah-cuan-credentials.json"

# Initialize the Google Cloud Storage client
storage_client = storage.Client()
bucket_name = "cuan-sampah-bucket"
bucket = storage_client.bucket(bucket_name)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Configure logging
logging.basicConfig(level=logging.INFO)

# Waste types data
jenis = [
    {'nama':'Kardus',
     'Jenis' : 'Organik',
     'Bahaya':
     '''
     Kardus merupakan jenis sampah Kertas yang dapat bercampur dengan sampah tipe lain, yaitu anorganik.
     Hal tersebut akan membuat pembusukan berlangsung tanpa oksigen atau anaerob. Pembusukan yang berlangsung secara anaerob itu akan menghasilkan gas metana.
     Gas metana dapat mempercepat perubahan iklim karena memiliki kekuatan menangkap panas di atmosfer bumi 25 kali lebih kuat dibandingkan karbondioksida.
     '''
     ,
     'Pengolahan':
     '''
     1. Sisa kardus bisa dimanfaatkan untuk mengemas dan menyimpan barang-barang selama kondisinya masih bagus dan tidak basah.
     2. Kardus dapat diolah menjadi karya seni. Pigura foto, organizer, celengan adalah contoh dari hasil kerajinan dari kardus bekas.
     3. Kardus bekas bisa dimanfaatkan untuk dijual kembali kepada para pengepul atau pengrajin yang membutuhkan bahan baku kardus.
     '''
    },
    {'nama': 'Kaca',
     'Jenis': 'Anorganik',
     'Bahaya':
     '''
     Limbah kaca yang tidak dikelola dengan benar dapat menyebabkan kemungkinan terlukanya masyarakat karena tersebarnya pecahan kaca di lingkungan dimana banyak terjadi aktivitas masyarakat seperti di area persawahan.
     Pembuangan limbah kaca secara sembarangan tidak hanya membahayakan masyarakat namun juga hewan hewan liar yang mencari makan atau tempat tinggal disekitarnya.
     '''
     ,
     'Pengolahan':
     '''
     Membawa limbah kaca ke pabrik atau tempat daur ulang, yang mana nantinya gelas akan dipecah menjadi potongan kecil-kecil yang disebut cullet.
     Potongan yang sudah dihancurkan, kemudian disortir dan dibersihkan, lalu dicampur bahan baku lain seperti soda abu dan pasir.
     '''
    },
    {'nama': 'Logam',
     'Jenis': 'Anorganik',
     'Bahaya':
     '''
     Pencemaran tanah dan air, membahayakan satwa liar, dan menurunkan kualitas air.
     Logam berat dalam limbah dapat meresap ke dalam rantai makanan, menyebabkan keracunan pada manusia dan hewan, serta berbagai penyakit kronis.
     Tepi tajam kaleng juga bisa menyebabkan luka fisik.
     '''
     ,
     'Pengolahan':
     '''
     1. Mengumpulkan limbah logam dan dipisahkan dengan sampah lainnya.
     2. Membersihkan logam dari kontaminan seperti plastik dan kertas, dan potong menjadi bagian kecil.
     3. Mendaur ulang logam
     '''
    },
    {'nama':'Kertas',
     'Jenis':'Organik',
     'Bahaya':
     '''
     sampah Kertas yang dapat bercampur dengan sampah tipe lain, yaitu anorganik.
     Hal tersebut akan membuat pembusukan berlangsung tanpa oksigen atau anaerob. Pembusukan yang berlangsung secara anaerob itu akan menghasilkan gas metana.
     Gas metana dapat mempercepat perubahan iklim karena memiliki kekuatan menangkap panas di atmosfer bumi 25 kali lebih kuat dibandingkan karbondioksida.
     '''
     ,
     'Pengolahan':
     '''
     1. Sisa kardus bisa dimanfaatkan untuk mengemas dan menyimpan barang-barang selama kondisinya masih bagus dan tidak basah.
     2. Kardus dapat diolah menjadi karya seni. Pigura foto, organizer, celengan adalah contoh dari hasil kerajinan dari kardus bekas.
     3. Kardus bekas bisa dimanfaatkan untuk dijual kembali kepada para pengepul atau pengrajin yang membutuhkan bahan baku kardus.
     '''
    },
    {'nama': 'Plastik',
     'Jenis': 'Anorganik',
     'Bahaya':
     '''
     Di lingkungan, plastik mencemari laut dan tanah, mengganggu ekosistem, dan dapat bertahan selama berabad-abad.
     Plastik juga dapat menyebabkan kontaminasi makanan dan minuman dengan mikroplastik, yang berpotensi merusak kesehatan manusia dengan bahan kimia berbahaya seperti BPA dan ftalat.
     '''
     ,
     'Pengolahan':
     '''
     1. Limbah plastik dikumpulkan secara terpisah dari jenis limbah lainnya untuk memudahkan proses selanjutnya.
     2. Plastik dipecah atau dilebur untuk diubah kembali menjadi bahan baku yang dapat digunakan kembali.
     3. Penggunaan teknologi seperti pirolisis untuk mengubah plastik menjadi bahan bakar alternatif atau produk kimia lainnya.
     '''
    },
]

vision_model = tf.keras.models.load_model('./myNewModel.h5')

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

# If your model needs image input use this endpoint!
@app.post("/predict_image")
async def predict_image(response: Response, uploaded_file: UploadFile=File(...)):
    try:
        logging.info(f"Received file: {uploaded_file.filename}")
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {"error": "File is Not an Image"}
        
        # Save the uploaded file to Google Cloud Storage
        blob = bucket.blob(f"uploads/{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.filename}")
        blob.upload_from_file(uploaded_file.file)
        file_url = blob.public_url
        logging.info(f"File uploaded to {file_url}")

        # Read image file and load it into PIL
        uploaded_file.file.seek(0)
        image = Image.open(uploaded_file.file)
        
        # Preprocess the image
        width, height = image.size
        left = (width - height) / 2
        right = left + height
        image = image.crop((left, 0, right, height))
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        logging.info(f"Image converted to numpy array. Shape: {image_array.shape}")

        image_array = np.expand_dims(image_array, axis=0)
        logging.info(f"Image shape with batch dimension: {image_array.shape}")

        processed_image = preprocess_input(image_array)
        logging.info(f"Processed shape: {processed_image.shape}")

        # Step 3: Predict the data
        result = vision_model.predict(processed_image, verbose=0)
        logging.info(f"Prediction result: {result}")
        result = np.round(result)

        if np.sum(result) == 1:
            kelas = np.where(result[0] == 1)
            logging.info(f"Predicted class index: {kelas}")

            jenis_sampah = jenis[kelas[0][0]]
            logging.info(f"Predicted class name: {jenis_sampah['nama']}")
            return {
                'nama': jenis_sampah['nama'],
                'Jenis Sampah': jenis_sampah['Jenis'],
                'Bahaya': jenis_sampah['Bahaya'],
                'Pengolahan': jenis_sampah['Pengolahan']
            }
        else:
            return {"message": "Hasil Identifikasi: Tidak Diketahui"}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": "Internal Server Error"}

port = int(os.environ.get("PORT", 8080))
print(f"Listening to http://localhost:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
