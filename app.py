import streamlit as st
import cv2
import face_recognition
import sqlite3
import numpy as np
from pyzbar import pyzbar
import pandas as pd

# --- 1. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('library.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY, 
                  name TEXT, 
                  face_encoding BLOB, 
                  current_issue TEXT DEFAULT 'No', 
                  barcode TEXT DEFAULT NULL)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. TRANSACTION LOGIC ---
def process_transaction(student_name, book_id):
    conn = sqlite3.connect('library.db')
    cur = conn.cursor()
    cur.execute("SELECT current_issue, barcode FROM students WHERE name=?", (student_name,))
    row = cur.fetchone()
    
    if row:
        status, existing_b = row
        if status == 'Yes':
            if existing_b == book_id:
                cur.execute("UPDATE students SET current_issue='No', barcode=NULL WHERE name=?", (student_name,))
                st.balloons()
                st.toast(f"✅ RETURNED: {book_id}")
            else:
                st.error(f"❌ {student_name} must return {existing_b} first!")
        else:
            cur.execute("UPDATE students SET current_issue='Yes', barcode=? WHERE name=?", (book_id, student_name))
            st.balloons()
            st.toast(f"📖 ISSUED: {book_id}")
        conn.commit()
    conn.close()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("📚 Library Menu")
st.sidebar.write("🟢 **System Status:** Online")
st.sidebar.write("🤖 **AI Core:** Active (128D)")
menu = st.sidebar.radio("Navigate to:", ["👤 Registration", "🛒 Smart Kiosk", "📊 View Records"])

st.sidebar.markdown("---")
st.sidebar.caption("🚀 **Project Credits**")
st.sidebar.write("👤 **Developer:** Shubham Shukla")
st.sidebar.write("🤖 **AI Architect:** Google Gemini")

# --- PAGE 1: REGISTRATION ---
if menu == "👤 Registration":
    st.title("Student Enrollment")
    st.info("Register once. The system blocks duplicate faces.")
    
    reg_name = st.text_input("Enter Full Name")
    img_file = st.camera_input("Capture Face")
    
    if st.button("Finalize Registration") and reg_name and img_file:
        image = face_recognition.load_image_file(img_file)
        encs = face_recognition.face_encodings(image)
        
        if encs:
            new_enc = encs[0]
            conn = sqlite3.connect('library.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name, face_encoding FROM students WHERE face_encoding IS NOT NULL")
            existing_data = cursor.fetchall()
            
            is_already_registered = False
            if existing_data:
                known_encs = [np.frombuffer(row[1], dtype=np.float64) for row in existing_data]
                matches = face_recognition.compare_faces(known_encs, new_enc, tolerance=0.4)
                if True in matches: is_already_registered = True

            if is_already_registered:
                st.error("🚨 DUPLICATE: You are already registered!")
            else:
                cursor.execute("INSERT INTO students (name, face_encoding) VALUES (?, ?)", 
                                (reg_name, new_enc.tobytes()))
                conn.commit()
                st.success(f"✅ {reg_name} registered!")
            conn.close()
        else:
            st.error("❌ No face detected.")

# --- PAGE 2: SMART KIOSK ---
elif menu == "🛒 Smart Kiosk":
    st.title("AI Automated Circulation Hub")
    
    if 'verified_user' not in st.session_state:
        st.session_state.verified_user = None

    col_face, col_book = st.columns([1.5, 1])

    with col_face:
        st.subheader("Step 1: Identify Face")
        run_face = st.toggle("Activate Face Scanner")
        FACE_WINDOW = st.image([]) 

        if run_face:
            conn = sqlite3.connect('library.db')
            rows = conn.cursor().execute("SELECT name, face_encoding FROM students WHERE face_encoding IS NOT NULL").fetchall()
            known_names = [r[0] for r in rows]
            known_encodings = [np.frombuffer(r[1], dtype=np.float64) for r in rows]
            conn.close()

            cap = cv2.VideoCapture(0)
            frame_count = 0
            while run_face:
                ret, frame = cap.read()
                if not ret: break
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                if frame_count % 5 == 0:
                    face_locations = face_recognition.face_locations(rgb_small, model="hog")
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                    for enc in face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.45)
                        if True in matches:
                            st.session_state.verified_user = known_names[matches.index(True)]

                FACE_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
                if st.session_state.verified_user: break 
            cap.release()

    with col_book:
        st.subheader("Step 2: Book Entry")
        if st.session_state.verified_user:
            st.success(f"Verified: {st.session_state.verified_user}")
            run_bar = st.toggle("Open Scanner")
            scanned_code = None
            if run_bar:
                BAR_WINDOW = st.empty()
                cap_bar = cv2.VideoCapture(0)
                while run_bar:
                    ret, frame = cap_bar.read()
                    if not ret: break
                    codes = pyzbar.decode(frame)
                    if codes:
                        scanned_code = codes[0].data.decode('utf-8')
                        break
                    BAR_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap_bar.release()

            manual_code = st.text_input("Enter 8-Digit Barcode")
            final_code = scanned_code if scanned_code else manual_code
            
            if st.button("Confirm Transaction") and final_code:
                if len(final_code) == 8 and final_code.isdigit():
                    process_transaction(st.session_state.verified_user, final_code)
                    st.session_state.verified_user = None
                    st.rerun()
                else:
                    st.error("Invalid Barcode (8 digits only)")
            
            if st.button("Logout"):
                st.session_state.verified_user = None
                st.rerun()
        else:
            st.warning("Verify face first.")

# --- PAGE 3: RECORDS ---
elif menu == "📊 View Records":
    st.title("Library Database")
    conn = sqlite3.connect('library.db')
    df = pd.read_sql_query("SELECT id, name, current_issue, barcode FROM students", conn)
    st.dataframe(df, use_container_width=True)
    conn.close()
