import mysql.connector
from mysql.connector import Error
import cv2
import datetime

def create_db_connection():
    try: 
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Lluvia190202*",
            database="peripheraldatabase"
        )
        return connection 
    except Error as e: 
        print(f"The error {e} ocurred")
        return None

def save_image_data(image_blob, classification, width, height, x_coord, y_coord, center_x, center_y, magnification, capture_date):
    connection = create_db_connection()
    if connection is not None:
        cursor = connection.cursor()
        query = """
        INSERT INTO cellimages (image, classification, width, height, x_coord, y_coord, center_x, center_y, magnification, capture_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (image_blob, classification, width, height, x_coord, y_coord, center_x, center_y, magnification, capture_date)
        try:
            cursor.execute(query, values)
            connection.commit()
            print("Image data saved successfully")
        except Error as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()
            connection.close()

