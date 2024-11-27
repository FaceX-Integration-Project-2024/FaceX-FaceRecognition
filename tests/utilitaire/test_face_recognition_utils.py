from unittest import TestCase
import numpy as np
from utilitaire.face_recognition_utils import *
from database.supabase_client import create_supabase_client
from config.env_loader import load_env_variables
from utilitaire.face_data_utils import checkFaceDataValidity

class Test_studentsImgToFaceData(TestCase):
    def setUp(self):
          """ Initialisation des variables de test"""
          env_vars = load_env_variables()
          self.supabase = create_supabase_client(env_vars['DB_URL'], env_vars['DB_KEY'])
          self.email = "gaetan.carbonnelle1@gmail.com"
          self.fake_email = "gaetan.cernauhfesl@gmail.com"
          self.matricule = ""


    def test_studentsImgToFaceData_success(self):

        face_data = studentsImgToFaceData(self.supabase, self.email)
        face_data = np.array(face_data)
        
        # Check visage existe  
        self.assertIsNotNone(face_data, 'Test_studentsImgToFaceData : Aucun visage détecté')
        
        # Check longeur data est juste
        self.assertTrue(len(face_data) == 128, 'Test_studentsImgToFaceData : Longeur face_data pas bonne')
        
        # Check si le tableau à les bonne données
        self.assertTrue(np.issubdtype(face_data.dtype, np.number), 'Test_studentsImgToFaceData : infos dans face_data incorect ')


    def test_studentsImgToFaceData_invalid_email(self):
         
        face_data = studentsImgToFaceData(self.supabase, self.fake_email)

        # Check s'il y a bien pas d'image
        self.assertIsNone(None,"Test_studentsImgToFaceData : il ne devrait pas y avoir d''image")


         
