# utils/database.py
from typing import List, Tuple
import sqlite3
import streamlit as st

DATABASE_NAME = "app_data.db"

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    return conn

def initialize_db():
    """Initializes the database by creating necessary tables with appropriate constraints."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create images table with UNIQUE constraint on image_title
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_title TEXT NOT NULL UNIQUE,
            image_data BLOB NOT NULL,
            prediction INTEGER,
            ground_truth INTEGER DEFAULT 0
        )
    ''')
    
    # Create CSV table with UNIQUE constraint on csv_name
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS csv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            csv_name TEXT NOT NULL UNIQUE,
            csv_data BLOB NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def clear_database():
    """Clears all data from the images and csv_data tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM images')
    cursor.execute('DELETE FROM csv_data')
    conn.commit()
    conn.close()

def fetch_images() -> List[Tuple]:
    """Fetches all images from the database along with their predictions and ground_truth values."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, image_title, image_data, prediction, ground_truth FROM images')
    images = cursor.fetchall()
    conn.close()
    return images

def insert_image(image_title: str, image_data: bytes, prediction: int = None, ground_truth: int = 0):
    """
    Inserts a new image into the database.
    If the image already exists, updates its prediction and ground_truth.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO images (image_title, image_data, prediction, ground_truth)
            VALUES (?, ?, ?, ?)
        ''', (image_title, image_data, prediction, ground_truth))
        conn.commit()
    except sqlite3.IntegrityError:
        # Image already exists, update prediction and ground_truth
        cursor.execute('''
            UPDATE images
            SET image_data = ?, prediction = ?, ground_truth = ?
            WHERE image_title = ?
        ''', (image_data, prediction, ground_truth, image_title))
        conn.commit()
    finally:
        conn.close()

def delete_image(image_title: str):
    """
    deletes image from app/db
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM images WHERE image_title = ?",
        (image_title,)
    )
    conn.commit()
    conn.close()

def insert_csv(csv_name: str, csv_data: bytes):
    """
    Inserts a new CSV into the database.
    If the CSV already exists, updates its data.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO csv_data (csv_name, csv_data)
            VALUES (?, ?)
        ''', (csv_name, csv_data))
        conn.commit()
    except sqlite3.IntegrityError:
        # CSV already exists, update csv_data
        cursor.execute('''
            UPDATE csv_data
            SET csv_data = ?
            WHERE csv_name = ?
        ''', (csv_data, csv_name))
        conn.commit()
    finally:
        conn.close()

def fetch_csv() -> List[Tuple]:
    """Fetches all CSVs from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, csv_name, csv_data FROM csv_data')
    csvs = cursor.fetchall()  # Corrected: Added parentheses
    conn.close()
    return csvs

def update_ground_truth(ground_truth: int, image_title: str):
    """Updates ground truth for a specific image."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE images SET ground_truth = ? WHERE image_title = ?",
        (ground_truth, image_title)
    )
    conn.commit()
    conn.close()

def update_image_prediction(image_title, prediction):
    """Helper function to update prediction in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE images SET prediction = ? WHERE image_title = ?",
            (prediction, image_title)
        )
        conn.commit()
        
        # Fetch the updated record to confirm
        cursor.execute(
            "SELECT prediction FROM images WHERE image_title = ?",
            (image_title,)
        )
        updated_prediction = cursor.fetchone()[0]
        # st.write(f"Database updated prediction for {image_title}: {updated_prediction}")  # debugging
    except Exception as e:
        st.error(f"Failed to update prediction for {image_title}: {e}")
    finally:
        conn.close()
